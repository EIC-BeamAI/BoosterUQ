using Test
using LinearAlgebra
using ChainRulesCore

include(joinpath(@__DIR__, "..", "src", "metal", "booster_batched_uq_metal.jl"))
using .BoosterBatchedUQMetal

const MUQ = BoosterBatchedUQMetal.UQ

function metal_test_points(nsettings)
    default = MUQ._BoosterLatticeTemplate.default_booster_operating_point()
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

@testset "device Metal factor-batched likelihood and gradient" begin
    nsettings = 2
    nfactors = 3
    prepared = MUQ.prepare_booster_batch(
        metal_test_points(nsettings),
        zeros(nsettings, MUQ.N_BPMS);
        noise_std_mm=0.2,
    )
    factor_sets = [
        1.0 + 0.0015sin(0.23factor + 0.17quad)
        for factor in 1:nfactors, quad in 1:MUQ.N_QUADS
    ]

    cpu = MUQ.BoosterFactorBatchProblem(
        prepared,
        nfactors;
        sensitivity=MUQ.BatchedForwardSensitivity(
            chunksize=4, warm_start=false, maxiter=40
        ),
    )
    cpu_values, cpu_gradients = MUQ.factor_batch_value_and_gradient!(
        cpu, factor_sets
    )

    metal = metal_factor_problem(
        prepared, nfactors; chunksize=4, warm_start=false, maxiter=40
    )
    metal_values, metal_gradients = metal_factor_value_and_gradient!(
        metal, factor_sets
    )

    @test size(metal_values) == (nfactors,)
    @test size(metal_gradients) == (nfactors, MUQ.N_QUADS)
    @test all(isfinite, metal_values)
    @test all(isfinite, metal_gradients)
    isolated_metal = metal_problem(
        prepared; chunksize=4, warm_start=false, maxiter=40
    )
    for factor in 1:nfactors
        @test metal_values[factor] ≈ cpu_values[factor] rtol=5e-4 atol=2e-3
        relative_gradient_error = norm(
            @view(metal_gradients[factor, :]) .-
            @view(cpu_gradients[factor, :])
        ) / norm(@view(cpu_gradients[factor, :]))
        @test relative_gradient_error < 0.12

        isolated_value, isolated_gradient = metal_value_and_gradient!(
            isolated_metal, @view factor_sets[factor, :]
        )
        @test metal_values[factor] ≈ isolated_value rtol=2e-6 atol=2e-4
        @test norm(@view(metal_gradients[factor, :]) - isolated_gradient) /
            norm(isolated_gradient) < 2e-5
    end

    rule_values, pullback = ChainRulesCore.rrule(
        metal_factor_batch_loglikelihoods, metal, factor_sets
    )
    weights = [0.5, -0.25, 1.25]
    _, _, rule_gradient = pullback(weights)
    @test rule_values ≈ metal_values
    @test rule_gradient ≈ reshape(weights, :, 1) .* metal_gradients
end

@testset "device Metal Booster likelihood and gradient" begin
    nsettings = 2
    prepared = MUQ.prepare_booster_batch(
        metal_test_points(nsettings),
        zeros(nsettings, MUQ.N_BPMS);
        noise_std_mm=0.2,
    )
    factors = 1 .+ 0.002 .* sin.(1:MUQ.N_QUADS)

    cpu = MUQ.BoosterBatchProblem(
        prepared;
        sensitivity=MUQ.BatchedForwardSensitivity(
            chunksize=4, warm_start=false, maxiter=40
        ),
    )
    cpu_value, cpu_gradient = MUQ.value_and_gradient!(cpu, factors)

    metal = metal_problem(prepared; chunksize=4, warm_start=false, maxiter=40)
    metal_value, metal_gradient = metal_value_and_gradient!(metal, factors)

    @test isfinite(metal_value)
    @test all(isfinite, metal_gradient)
    # Metal uses Float32 because current Apple GPUs do not provide Float64.
    @test metal_value ≈ cpu_value rtol=5e-4 atol=2e-3
    # Elementwise relative error is misleading for components close to zero;
    # compare the shared-gradient vector as a whole.
    @test norm(metal_gradient - cpu_gradient) / norm(cpu_gradient) < 0.08
end
