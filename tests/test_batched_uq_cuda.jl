using Test
using ChainRulesCore
using LinearAlgebra
using CUDA
using SciBmad

CUDA.functional() || error(
    "CUDA is not functional. Run `CUDA.versioninfo()` and resolve the driver/device issue first."
)
CUDA.allowscalar(false)

println("CUDA device: ", CUDA.device())
println("CUDA_VISIBLE_DEVICES: ", get(ENV, "CUDA_VISIBLE_DEVICES", "unset"))
CUDA.versioninfo()

include(joinpath(@__DIR__, "..", "src", "cuda", "booster_batched_uq_cuda.jl"))
using .BoosterBatchedUQCUDA

const CUQ = BoosterBatchedUQCUDA.UQ
const CUDA_TEST_CHUNK = parse(Int, get(ENV, "BOOSTER_CUDA_TEST_CHUNK", "4"))
const CUDA_TEST_TYPE_NAME = get(ENV, "BOOSTER_CUDA_TEST_TYPE", "Float32")
const CUDA_TEST_TYPE = if CUDA_TEST_TYPE_NAME == "Float32"
    Float32
elseif CUDA_TEST_TYPE_NAME == "Float64"
    Float64
else
    error("BOOSTER_CUDA_TEST_TYPE must be Float32 or Float64")
end

const CUDA_CONVERT = isdefined(CUDA, :cudaconvert) ?
    CUDA.cudaconvert : CUDA.CUDACore.cudaconvert

function cuda_test_points(nsettings)
    default = CUQ._BoosterLatticeTemplate.default_booster_operating_point()
    [
        merge(default, (;
            idipo=default.idipo * (1 + 0.01sin(0.31setting)),
            iqhc=default.iqhc * (1 + 0.006cos(0.27setting)),
            iqvc=default.iqvc * (1 + 0.006sin(0.23setting)),
            ish=default.ish + 0.2cos(0.19setting),
            isv=default.isv + 0.2sin(0.17setting),
            bdot=default.bdot + 0.02sin(0.29setting),
            corrector_currents=ntuple(
                corrector -> 0.1sin(0.13setting + 0.07corrector), 48
            ),
        ))
        for setting in 1:nsettings
    ]
end

@testset "CUDA kernel argument adaptation" begin
    values = CuArray(CUDA_TEST_TYPE[1, 2])
    lowered = CUQ.BeamTracking._LoweredBatchParam(values)
    converted = CUDA_CONVERT(lowered)
    @test isbitstype(typeof(converted))

    reference = CUQ.BeamTracking.RefState{CUDA_TEST_TYPE}(
        t_enter=zero(CUDA_TEST_TYPE),
        beta_gamma_enter=lowered,
        t_exit=lowered,
        beta_gamma_exit=lowered,
    )
    converted_reference = CUDA_CONVERT(reference)
    @test isbitstype(typeof(converted_reference))
end

@testset "DHCD6 and DHCF6 corrector mapping" begin
    base = CUQ._BoosterLatticeTemplate.default_booster_operating_point()
    point(index) = merge(base, (;
        corrector_currents=ntuple(
            corrector -> corrector == index ? 1.0 : 0.0,
            length(CUQ.CORRECTOR_NAMES),
        ),
    ))
    prepared = CUQ.prepare_booster_batch(
        [point(0), point(15), point(23)],
        zeros(3, CUQ.N_BPMS);
        noise_std_mm=0.2,
    )
    @test !iszero(prepared.hcorrector_k0L[2, 15])
    @test !iszero(prepared.hcorrector_k0L[3, 23])

    cpu = CUQ.BoosterBatchProblem(
        prepared;
        sensitivity=CUQ.BatchedForwardSensitivity(
            chunksize=4, warm_start=false, maxiter=40
        ),
    )
    orbit = CUQ.predict_orbits(cpu, ones(CUQ.N_QUADS))
    @test norm(@view(orbit[2, :]) - @view(orbit[1, :])) > 0
    @test norm(@view(orbit[3, :]) - @view(orbit[1, :])) > 0
end

@testset "CUDA Booster likelihood and gradient" begin
    nsettings = 2
    prepared = CUQ.prepare_booster_batch(
        cuda_test_points(nsettings),
        zeros(nsettings, CUQ.N_BPMS);
        noise_std_mm=0.2,
    )
    factors = 1 .+ 0.002 .* sin.(1:CUQ.N_QUADS)

    println("Constructing CPU reference problem")
    cpu = CUQ.BoosterBatchProblem(
        prepared;
        sensitivity=CUQ.BatchedForwardSensitivity(
            chunksize=CUDA_TEST_CHUNK, warm_start=false, maxiter=40
        ),
    )
    cpu_value, cpu_gradient = CUQ.value_and_gradient!(cpu, factors)

    println(
        "Constructing CUDA problem: eltype=", CUDA_TEST_TYPE,
        " chunksize=", CUDA_TEST_CHUNK,
    )
    gpu = cuda_problem(
        prepared;
        chunksize=CUDA_TEST_CHUNK,
        device_eltype=CUDA_TEST_TYPE,
        warm_start=false,
        maxiter=40,
    )

    # Stage 1 isolates the device Newton solve and fixed-point Jacobian.
    println("Stage 1/3: on-device Newton solve and fixed-point matrix")
    BoosterBatchedUQCUDA._update_device_primal!(gpu, factors)
    BoosterBatchedUQCUDA._solve_device_orbit!(gpu)
    BoosterBatchedUQCUDA._update_device_primal!(gpu, factors)
    gpu_orbit = Array(gpu.device.primal.closed_orbit)
    gpu_lhs = Array(gpu.device.primal.fixed_point_lhs)
    @test all(isfinite, gpu_orbit)
    @test all(isfinite, gpu_lhs)
    tracked_orbit = copy(gpu.device.primal.closed_orbit)
    CUQ._track!(tracked_orbit, gpu.device.primal.lattice)
    orbit_residual = maximum(abs, Array(tracked_orbit)[:, 1:4] .- gpu_orbit[:, 1:4])
    orbit_error = maximum(abs, gpu_orbit .- cpu.primal.closed_orbit)
    lhs_error = maximum(abs, gpu_lhs .- cpu.primal.fixed_point_lhs)
    println("closed-orbit maximum residual: ", orbit_residual)
    println("closed-orbit maximum CPU difference: ", orbit_error)
    println("I-R maximum CPU difference: ", lhs_error)
    @test orbit_residual < (CUDA_TEST_TYPE === Float32 ? 5e-5 : 1e-9)
    @test lhs_error < (CUDA_TEST_TYPE === Float32 ? 2e-3 : 2e-6)
    expected_quadrupoles = CUDA_TEST_TYPE.(prepared.quad_k1_base) .*
        reshape(CUDA_TEST_TYPE.(factors), 1, :)
    @test Array(gpu.device.primal.quad_active) ≈ expected_quadrupoles

    # Stage 2 is primal CUDA tracking plus the likelihood reduction, before any
    # ForwardDiff-dual kernels are compiled.
    println("Stage 2/3: primal CUDA tracking and likelihood reduction")
    gpu_value_only = Array(BoosterBatchedUQCUDA._device_prediction_and_weights!(gpu))[1]
    CUDA.synchronize()
    @test isfinite(gpu_value_only)
    prediction_difference = Array(gpu.device.primal.predicted_orbit_mm) .-
        cpu.primal.predicted_orbit_mm
    prediction_error = norm(prediction_difference) /
        max(norm(cpu.primal.predicted_orbit_mm), eps())
    prediction_rms_error_mm = norm(prediction_difference) /
        sqrt(length(prediction_difference))
    prediction_max_error_mm = maximum(abs, prediction_difference)
    println("prediction relative L2 error: ", prediction_error)
    println("prediction RMS error [mm]: ", prediction_rms_error_mm)
    println("prediction max error [mm]: ", prediction_max_error_mm)

    # Stage 3 compiles and exercises the implicit solve and tangent tracker.
    println("Stage 3/3: complete CUDA likelihood and gradient")
    gpu_value, gpu_gradient = cuda_value_and_gradient!(gpu, factors)
    native_value, native_gradient = value_and_gradient!(
        gpu, CuArray(CUDA_TEST_TYPE.(factors)),
    )
    native_status = device_status(gpu)
    @test native_value isa CuArray
    @test native_gradient isa CuArray
    @test all(Array(native_status.converged))
    @test !any(Array(native_status.orbit_failed))
    @test !any(Array(native_status.sensitivity_failed))
    @test !any(Array(native_status.input_invalid))
    @test !any(Array(native_status.tracking_lost))
    if CUDA_TEST_TYPE === Float64
        sci_lhs_error = maximum(abs, Array(gpu.device.primal.fixed_point_lhs) .- gpu_lhs)
        println("SciBmad I-R versus custom device I-R maximum difference: ", sci_lhs_error)
        @test sci_lhs_error < 2e-6
    end
    @test isapprox(Array(native_value)[1], gpu_value; rtol=1e-6, atol=1e-4)
    @test norm(vec(Array(native_gradient)) .- gpu_gradient) /
        max(norm(gpu_gradient), eps()) < 2e-4
    native_rule_value, native_pullback = ChainRulesCore.rrule(
        cuda_booster_loglikelihood, gpu, CuArray(CUDA_TEST_TYPE.(factors)),
    )
    _, _, native_rule_gradient = native_pullback(CuArray(CUDA_TEST_TYPE[1]))
    @test native_rule_value isa CuArray
    @test native_rule_gradient isa CuArray
    @test norm(Array(native_rule_gradient) .- gpu_gradient) /
        max(norm(gpu_gradient), eps()) < 2e-4
    @test cuda_value_and_gradient! === CUQ.value_and_gradient!
    @test cuda_booster_loglikelihood === CUQ.booster_loglikelihood
    @test isfinite(gpu_value)
    @test all(isfinite, gpu_gradient)

    value_error = abs(gpu_value - cpu_value) / max(abs(cpu_value), 1.0)
    gradient_error = norm(gpu_gradient - cpu_gradient) / norm(cpu_gradient)
    gradient_cosine = dot(gpu_gradient, cpu_gradient) /
        (norm(gpu_gradient) * norm(cpu_gradient))
    println("likelihood relative error: ", value_error)
    cpu_data_term = prepared.loglik_constant - cpu_value
    gpu_data_term = prepared.loglik_constant - gpu_value
    data_term_error = abs(gpu_data_term - cpu_data_term) /
        max(abs(cpu_data_term), eps())
    println("likelihood data-term relative error: ", data_term_error)
    println("gradient relative L2 error: ", gradient_error)
    println("gradient cosine similarity: ", gradient_cosine)

    # Compare against the derivative of the actual CUDA objective. A
    # gradient-aligned direction avoids a near-zero directional derivative and
    # the resulting catastrophic Float32 cancellation.
    direction = gpu_gradient ./ norm(gpu_gradient)
    step = CUDA_TEST_TYPE === Float32 ? 2e-2 : 2e-5
    plus = factors .+ step .* direction
    minus = factors .- step .* direction
    finite_difference = (
        cuda_loglikelihood_value!(gpu, plus) -
        cuda_loglikelihood_value!(gpu, minus)
    ) / (2step)
    analytic_directional = dot(gpu_gradient, direction)
    directional_error = abs(analytic_directional - finite_difference) /
        max(abs(finite_difference), abs(analytic_directional), eps())
    println("CUDA directional derivative: ", analytic_directional)
    println("CUDA finite difference: ", finite_difference)
    println("CUDA directional relative error: ", directional_error)

    if CUDA_TEST_TYPE === Float32
        @test prediction_rms_error_mm < 2e-2
        @test value_error < 5e-3
        @test gradient_error < 1.5e-1
        @test gradient_cosine > 0.99
        @test directional_error < 1e-1
    else
        @test prediction_error < 5e-8
        @test value_error < 5e-8
        @test gradient_error < 5e-6
        @test gradient_cosine > 1 - 1e-8
        @test directional_error < 1e-4
    end

    rule_value, pullback = ChainRulesCore.rrule(
        cuda_booster_loglikelihood, gpu, factors
    )
    _, _, rule_gradient = pullback(1.0)
    @test rule_value ≈ gpu_value
    @test rule_gradient ≈ gpu_gradient
end

@testset "CUDA factor batch, isolated rows, and independent chains" begin
    nsettings, nfactors = 2, 3
    prepared = CUQ.prepare_booster_batch(
        cuda_test_points(nsettings), zeros(nsettings, CUQ.N_BPMS);
        noise_std_mm=0.2,
    )
    factors = [
        1.0 + 0.0015sin(0.23factor + 0.17quad)
        for factor in 1:nfactors, quad in 1:CUQ.N_QUADS
    ]
    cpu = CUQ.BoosterFactorBatchProblem(
        prepared, nfactors;
        sensitivity=CUQ.BatchedForwardSensitivity(
            chunksize=CUDA_TEST_CHUNK, warm_start=false, maxiter=40,
        ),
    )
    cpu_values, cpu_gradients = CUQ.factor_batch_value_and_gradient!(cpu, factors)
    gpu = cuda_factor_problem(
        prepared, nfactors; chunksize=CUDA_TEST_CHUNK,
        device_eltype=CUDA_TEST_TYPE, warm_start=false, maxiter=40,
    )
    println("Factor batch: complete CUDA likelihood and gradient")
    values, gradients = cuda_factor_value_and_gradient!(gpu, factors)
    native_values, native_gradients = factor_batch_value_and_gradient!(
        gpu, CuArray(CUDA_TEST_TYPE.(factors)),
    )
    native_status = device_status(gpu)
    @test native_values isa CuArray
    @test native_gradients isa CuArray
    @test all(Array(native_status.converged))
    @test !any(Array(native_status.orbit_failed))
    @test !any(Array(native_status.sensitivity_failed))
    @test !any(Array(native_status.input_invalid))
    @test !any(Array(native_status.tracking_lost))
    @test Array(native_values) ≈ values
    @test norm(Array(native_gradients) .- gradients) /
        max(norm(gradients), eps()) < 2e-4
    native_rule_values, native_pullback = ChainRulesCore.rrule(
        cuda_factor_batch_loglikelihoods, gpu, CuArray(CUDA_TEST_TYPE.(factors)),
    )
    native_weights = CuArray(CUDA_TEST_TYPE[0.5, -0.25, 1.25])
    _, _, native_rule_gradient = native_pullback(native_weights)
    @test native_rule_values isa CuArray
    @test native_rule_gradient isa CuArray
    @test Array(native_rule_gradient) ≈
        reshape(Array(native_weights), :, 1) .* gradients
    @test cuda_factor_value_and_gradient! === CUQ.factor_batch_value_and_gradient!
    @test cuda_factor_batch_loglikelihoods === CUQ.booster_factor_batch_loglikelihoods
    @test size(values) == (nfactors,)
    @test size(gradients) == (nfactors, CUQ.N_QUADS)
    @test all(isfinite, values)
    @test all(isfinite, gradients)
    batch_orbit = Array(gpu.device.primal.closed_orbit)
    batch_lhs = Array(gpu.device.primal.fixed_point_lhs)
    single = cuda_problem(
        prepared; chunksize=CUDA_TEST_CHUNK, device_eltype=CUDA_TEST_TYPE,
        warm_start=false, maxiter=40,
    )
    for factor in 1:nfactors
        row_factors = @view factors[factor, :]
        single_value, single_gradient = cuda_value_and_gradient!(single, row_factors)
        first_row = (factor - 1) * nsettings + 1
        rows = first_row:(first_row + nsettings - 1)
        value_error = abs(values[factor] - cpu_values[factor]) /
            max(abs(cpu_values[factor]), 1.0)
        gradient_error = norm(gradients[factor, :] .- cpu_gradients[factor, :]) /
            max(norm(cpu_gradients[factor, :]), eps())
        orbit_parity = maximum(abs, batch_orbit[rows, :] .-
            Array(single.device.primal.closed_orbit))
        lhs_parity = maximum(abs, batch_lhs[rows, :, :] .-
            Array(single.device.primal.fixed_point_lhs))
        println("factor $factor: CPU value error=$value_error, CPU gradient error=$gradient_error, orbit parity=$orbit_parity, I-R parity=$lhs_parity")
        @test value_error < (CUDA_TEST_TYPE === Float32 ? 5e-3 : 5e-8)
        @test gradient_error < (CUDA_TEST_TYPE === Float32 ? 0.15 : 5e-6)
        @test isapprox(values[factor], single_value; rtol=1e-6, atol=1e-4)
        @test norm(gradients[factor, :] .- single_gradient) /
            max(norm(single_gradient), eps()) < 2e-4
        @test orbit_parity < (CUDA_TEST_TYPE === Float32 ? 1e-5 : 1e-10)
        @test lhs_parity < (CUDA_TEST_TYPE === Float32 ? 1e-3 : 1e-8)
    end

    rule_values, pullback = ChainRulesCore.rrule(
        cuda_factor_batch_loglikelihoods, gpu, factors,
    )
    weights = [0.5, -0.25, 1.25]
    _, _, rule_gradient = pullback(weights)
    @test rule_values ≈ values
    @test rule_gradient ≈ reshape(weights, :, 1) .* gradients

    direction = gradients[1, :] ./ norm(gradients[1, :])
    step = CUDA_TEST_TYPE === Float32 ? 2e-2 : 2e-5
    plus, minus = copy(factors), copy(factors)
    plus[1, :] .+= step .* direction
    minus[1, :] .-= step .* direction
    finite_difference = (
        cuda_factor_loglikelihood_values!(gpu, plus)[1] -
        cuda_factor_loglikelihood_values!(gpu, minus)[1]
    ) / (2step)
    analytic = dot(gradients[1, :], direction)
    directional_error = abs(analytic - finite_difference) /
        max(abs(analytic), abs(finite_difference), eps())
    println("factor directional derivative error: ", directional_error)
    @test directional_error < (CUDA_TEST_TYPE === Float32 ? 0.1 : 1e-4)

    chains = cuda_chain_problems(
        prepared; nchains=2, chunksize=CUDA_TEST_CHUNK,
        device_eltype=CUDA_TEST_TYPE, warm_start=false, maxiter=40,
    )
    @test chains[1] !== chains[2]
    @test chains[1].device.primal.closed_orbit !== chains[2].device.primal.closed_orbit
    chain_factors = (copy(@view(factors[1, :])), copy(@view(factors[2, :])))
    chain_results = Vector{Any}(undef, 2)
    chain_threads = zeros(Int, 2)
    Threads.@threads :static for i in 1:2
        chain_threads[i] = Threads.threadid()
        chain_results[i] = cuda_value_and_gradient!(chains[i], chain_factors[i])
    end
    @test length(unique(chain_threads)) == 2
    for i in 1:2
        @test isapprox(chain_results[i][1], values[i]; rtol=1e-6, atol=1e-4)
        @test norm(chain_results[i][2] .- gradients[i, :]) /
            max(norm(gradients[i, :]), eps()) < 2e-4
    end

    warm = cuda_problem(
        prepared; chunksize=CUDA_TEST_CHUNK, device_eltype=CUDA_TEST_TYPE,
        warm_start=true, maxiter=40,
    )
    cuda_loglikelihood_value!(warm, chain_factors[1])
    warm_value = cuda_loglikelihood_value!(warm, chain_factors[2])
    @test isfinite(warm_value)
    if CUDA_TEST_TYPE === Float64
        @test isapprox(warm_value, values[2]; rtol=1e-8, atol=1e-4)
    end
    @test maximum(abs, Array(warm.device.primal.closed_orbit_guess) .-
        Array(warm.device.primal.closed_orbit)) == 0
end

@testset "SciBmad CUDA closed-orbit comparison" begin
    nsettings = 2
    prepared = CUQ.prepare_booster_batch(
        cuda_test_points(nsettings), zeros(nsettings, CUQ.N_BPMS);
        noise_std_mm=0.2,
    )
    factors = 1 .+ 0.002 .* sin.(1:CUQ.N_QUADS)
    cpu_lattice = CUQ.BoosterBatchProblem(prepared).primal.lattice
    check_reference = Base.get_extension(
        CUQ.BeamTracking, :BeamTrackingBeamlinesExt,
    ).check_bl_bunch!
    try
        check_reference(CUQ.BeamTracking.Bunch(zeros(nsettings, 6)), cpu_lattice, false)
        @info "Default CPU Bunch now accepts the batched reference"
    catch err
        err isa TypeError || rethrow()
        @info "Default CPU Bunch has a scalar reference field" error=sprint(showerror, err)
    end
    batch_reference = cpu_lattice.p_over_q_ref
    @test check_reference(
        CUQ.BeamTracking.Bunch(zeros(nsettings, 6); p_over_q_ref=batch_reference),
        cpu_lattice, false,
    )[2] isa CUQ.BatchParam
    @test check_reference(
        CUQ.BeamTracking.Bunch(
            zeros(nsettings, 6); p_over_q_ref=batch_reference,
            t_ref=CUQ.BatchParam(0.0),
        ),
        cpu_lattice, false,
    )[2] isa CUQ.BatchParam
    cpu_scibmad = SciBmad.find_closed_orbit(
        cpu_lattice; v0=zeros(nsettings, 6),
        coasting_beam=true, batch=Val(true), rf_on=false, warn=false,
        abstol=1e-11, reltol=0.0, maxiter=40,
    )
    @test all(==(SciBmad.RETCODE_SUCCESS), cpu_scibmad.sol.retcode)
    cpu_reference = CUQ.BoosterBatchProblem(prepared)
    CUQ._solve_updated_primal!(cpu_reference)
    for lane in 1:nsettings, row in 1:4, column in 1:4
        index = row + 4 * (lane - 1) + 4 * nsettings * (column - 1)
        @test isapprox(
            cpu_scibmad.sol.jac.nzval[index],
            cpu_reference.primal.fixed_point_lhs[lane, row, column]; atol=1e-8,
        )
    end
    gpu = cuda_problem(
        prepared; chunksize=CUDA_TEST_CHUNK, device_eltype=CUDA_TEST_TYPE,
        warm_start=false, maxiter=40,
    )
    BoosterBatchedUQCUDA._update_device_primal!(gpu, factors)
    BoosterBatchedUQCUDA._solve_device_orbit!(gpu)
    custom_orbit = Array(gpu.device.primal.closed_orbit)
    scibmad = SciBmad.find_closed_orbit(
        gpu.device.primal.lattice;
        v0=copy(gpu.device.primal.closed_orbit_guess),
        coasting_beam=true, batch=Val(true), rf_on=false, warn=false,
        abstol=CUDA_TEST_TYPE === Float32 ? 1e-5 : 1e-11,
        reltol=0.0, maxiter=40,
    )
    sci_orbit = Array(scibmad.v0)
    @test all(isfinite, sci_orbit)
    @test all(==(SciBmad.RETCODE_SUCCESS), Array(scibmad.sol.retcode))
    difference = maximum(abs, sci_orbit .- custom_orbit)
    println("SciBmad versus custom CUDA orbit maximum difference: ", difference)
    @test difference < (CUDA_TEST_TYPE === Float32 ? 1e-4 : 1e-8)
end

if CUDA_TEST_TYPE === Float64 && get(ENV, "BOOSTER_CUDA_TEST_LARGE_LHS", "0") == "1"
    @testset "SciBmad 32K-lane CUDA I-R" begin
        nexperiments = parse(Int, get(ENV, "BOOSTER_CUDA_TEST_LARGE_EXPERIMENTS", "128"))
        nfactors = 32768 ÷ nexperiments
        @test nexperiments * nfactors == 32768
        prepared = CUQ.prepare_booster_batch(
            cuda_test_points(nexperiments), zeros(nexperiments, CUQ.N_BPMS);
            noise_std_mm=0.2,
        )
        problem = cuda_factor_problem(
            prepared, nfactors; chunksize=4, device_eltype=Float64,
            warm_start=false, maxiter=40,
        )
        factors = CuArray([
            1.0 + 0.002sin(0.071factor + 0.113quad)
            for factor in 1:nfactors, quad in 1:CUQ.N_QUADS
        ])
        CUQ._prepare_factors!(problem, factors)
        BoosterBatchedUQCUDA._solve_scibmad_orbit!(problem)
        @test all(Array(device_status(problem).converged))
        work, primal = problem.orbit, problem.device.primal
        backend = CUQ.KernelAbstractions.get_backend(work.state)
        BoosterBatchedUQCUDA._seed_orbit!(backend)(
            work.state, primal.closed_orbit; ndrange=size(work.state),
        )
        CUQ._track!(work.state, primal.lattice; threaded=false)
        BoosterBatchedUQCUDA._orbit_linearization!(backend)(
            work.lhs, work.rhs, work.state, primal.closed_orbit;
            ndrange=size(work.lhs, 1),
        )
        residual = maximum(abs.(work.rhs))
        lhs_error = maximum(abs.(work.lhs .- primal.fixed_point_lhs))
        println("32K-lane $(nexperiments)x$(nfactors) residual=$residual I-R error=$lhs_error")
        @test residual < 1e-10
        @test lhs_error < 1e-8
    end
end
