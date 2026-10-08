using CUDA, Dates, LinearAlgebra, Printf, Statistics

CUDA.functional() || error("CUDA is not functional")
CUDA.allowscalar(false)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "..", "src", "cuda", "booster_batched_uq_cuda.jl"))
using .BoosterBatchedUQCUDA
const UQ = BoosterBatchedUQCUDA.UQ

const PAIRS = map(split(get(ENV, "BOOSTER_INFERENCE_PAIRS", "1:1"), ',')) do pair
    n, f = parse.(Int, split(pair, ':'))
    n > 0 && f > 0 && n * f <= 32768 || error("invalid experiments:factors pair $pair")
    (n, f)
end
const SAMPLES = parse(Int, get(ENV, "BOOSTER_INFERENCE_SAMPLES", "3"))
const OUTPUT = abspath(get(ENV, "BOOSTER_INFERENCE_OUTPUT", "cuda_inference.csv"))
const PROFILE = get(ENV, "BOOSTER_INFERENCE_PROFILE", "0") == "1"
const PROFILE_PAIR = get(ENV, "BOOSTER_INFERENCE_PROFILE_PAIR", "1:1")
SAMPLES >= 2 || error("at least two timed samples are required")

function operating_points(n)
    base = UQ._BoosterLatticeTemplate.default_booster_operating_point()
    [merge(base, (;
        idipo=base.idipo * (1 + 0.01sin(0.031i)),
        iqhc=base.iqhc * (1 + 0.01cos(0.037i)),
        iqvc=base.iqvc * (1 + 0.01sin(0.041i)),
        corrector_currents=ntuple(q -> 0.1sin(0.013i + 0.07q), UQ.N_QUADS),
    )) for i in 1:n]
end

factor_sets(n, sample) = [
    1.0 + 0.002sin(0.071f + 0.113q) + 0.00005sin(0.3sample + 0.19f - 0.07q)
    for f in 1:n, q in 1:UQ.N_QUADS
]
device_factors(n, sample) = CuArray(n == 1 ? vec(factor_sets(n, sample)) : factor_sets(n, sample))

function check_result(problem, result, mode)
    values, gradient = mode == :value_gradient ? result : (result, nothing)
    v = Array(values)
    all(isfinite, v) || error("nonfinite likelihood")
    g = gradient === nothing ? nothing : Array(gradient)
    g === nothing || all(isfinite, g) || error("nonfinite gradient")
    status = device_status(problem)
    all(Array(status.converged)) || error("closed orbit did not converge")
    any(Array(status.orbit_failed)) && error("closed-orbit Jacobian failed")
    any(Array(status.tracking_lost)) && error("particle lost")
    any(Array(status.input_invalid)) && error("invalid factor")
    any(Array(status.sensitivity_failed)) && error("implicit sensitivity failed")
    sum(v), g === nothing ? NaN : norm(g)
end

evaluate(problem, factors, mode) = mode == :value_gradient ?
    value_and_gradient!(problem, factors) : loglikelihood_value!(problem, factors)

function timed_call(problem, factors, mode; profile=false)
    CUDA.synchronize()
    start = time_ns()
    if profile
        captured = Ref{Any}()
        CUDA.@profile external=true begin
            captured[] = evaluate(problem, factors, mode)
            CUDA.synchronize()
        end
        result = captured[]
    else
        result = evaluate(problem, factors, mode)
    end
    CUDA.synchronize()
    seconds = (time_ns() - start) / 1e9
    value_sum, gradient_norm = check_result(problem, result, mode)
    seconds, value_sum, gradient_norm
end

mkpath(dirname(OUTPUT))
println("CUDA device=$(CUDA.device()) Float64 width=4 pairs=$(PAIRS) samples=$SAMPLES")
println("Scope: device input through device output, including SciBmad orbit, BPM tracking, implicit response and reductions")
failed = false
open(OUTPUT, "w") do io
    println(io, "gpu,experiments,factors,lanes,mode,sample,seconds,value_sum,gradient_norm,status,error")
    for (nexperiments, nfactors) in PAIRS
        lanes = nexperiments * nfactors
        prefix = "$(get(ENV, "CUDA_VISIBLE_DEVICES", "unset")),$nexperiments,$nfactors,$lanes"
        println("Preparing experiments=$nexperiments factors=$nfactors lanes=$lanes")
        try
            setup_start = time_ns()
            prepared = UQ.prepare_booster_batch(
                operating_points(nexperiments), zeros(nexperiments, UQ.N_BPMS);
                noise_std_mm=0.2,
            )
            problem = nfactors == 1 ?
                cuda_problem(prepared; chunksize=4, device_eltype=Float64,
                    warm_start=true, maxiter=40) :
                cuda_factor_problem(prepared, nfactors; chunksize=4,
                    device_eltype=Float64, warm_start=true, maxiter=40)
            CUDA.synchronize()
            setup_seconds = (time_ns() - setup_start) / 1e9
            @printf(io, "%s,setup,0,%.9f,NaN,NaN,ok,\n", prefix, setup_seconds)
            flush(io)

            for mode in (:value_gradient, :likelihood)
                warm_factors = device_factors(nfactors, mode == :value_gradient ? 0 : 1000)
                warm_seconds, _, _ = timed_call(problem, warm_factors, mode)
                @printf(io, "%s,%s_warmup,0,%.9f,NaN,NaN,ok,\n", prefix, mode, warm_seconds)
                flush(io)
                if PROFILE && mode == :value_gradient &&
                    PROFILE_PAIR == "$nexperiments:$nfactors"
                    seconds, value_sum, gradient_norm = timed_call(
                        problem, warm_factors, mode; profile=true)
                    @printf(io, "%s,value_gradient_profile,0,%.9f,%.17g,%.17g,ok,\n",
                        prefix, seconds, value_sum, gradient_norm)
                    flush(io)
                end
                elapsed = Float64[]
                for sample in 1:SAMPLES
                    factors = device_factors(
                        nfactors, sample + (mode == :value_gradient ? 0 : 1000))
                    seconds, value_sum, gradient_norm = timed_call(problem, factors, mode)
                    push!(elapsed, seconds)
                    @printf(io, "%s,%s,%d,%.9f,%.17g,%.17g,ok,\n",
                        prefix, mode, sample, seconds, value_sum, gradient_norm)
                    flush(io)
                    @printf("lanes=%d mode=%s sample=%d time=%.4fs\n",
                        lanes, mode, sample, seconds)
                end
                @printf("lanes=%d mode=%s median=%.4fs factors/s=%.2f lanes/s=%.2f\n",
                    lanes, mode, median(elapsed), nfactors / median(elapsed),
                    lanes / median(elapsed))
            end
            problem = nothing
            GC.gc(true)
            CUDA.reclaim()
        catch err
            failed = true
            message = replace(sprint(showerror, err), '\n' => ' ', '"' => "''")
            println(io, "$prefix,error,0,NaN,NaN,NaN,error,\"$message\"")
            flush(io)
            @error "Inference benchmark pair failed" nexperiments nfactors exception=(err, catch_backtrace())
        end
    end
end
failed && exit(1)
