using Printf
using Statistics
using LinearAlgebra
using Metal

include(joinpath(@__DIR__, "..", "src", "metal", "booster_batched_uq_metal.jl"))
using .BoosterBatchedUQMetal

const MUQ = BoosterBatchedUQMetal.UQ
const BATCH_SIZES = parse.(
    Int, split(get(ENV, "BOOSTER_METAL_BATCHES", "1,16,64,256,1024"), ',')
)
const CHUNKSIZES = parse.(
    Int, split(get(ENV, "BOOSTER_METAL_CHUNKS", "8"), ',')
)
const SAMPLES = parse(Int, get(ENV, "BOOSTER_METAL_SAMPLES", "3"))
const OUTPUT = get(
    ENV,
    "BOOSTER_METAL_OUTPUT",
    joinpath(@__DIR__, "benchmark_results", "booster_uq_metal.csv"),
)

function operating_points(nsettings)
    default = MUQ._BoosterLatticeTemplate.default_booster_operating_point()
    [
        merge(default, (;
            idipo=default.idipo * (1 + 0.02sin(0.031setting)),
            iqhc=default.iqhc * (1 + 0.02cos(0.037setting)),
            iqvc=default.iqvc * (1 + 0.02sin(0.041setting)),
            ish=default.ish * (1 + 0.03cos(0.043setting)),
            isv=default.isv * (1 + 0.03sin(0.047setting)),
            bdot=default.bdot + 0.05sin(0.029setting),
            corrector_currents=ntuple(
                corrector ->
                    0.35sin(0.19mod1(setting, 64) + 0.13corrector) +
                    0.08cos(0.07mod1(setting, 31) - 0.17corrector),
                48,
            ),
        ))
        for setting in 1:nsettings
    ]
end

function timed_samples(f)
    times = Float64[]
    for _ in 1:SAMPLES
        GC.gc()
        push!(times, @elapsed f())
    end
    minimum(times), median(times)
end

mkpath(dirname(OUTPUT))
open(OUTPUT, "w") do io
    println(
        io,
        "batch_size,chunksize,cpu_min_seconds,cpu_median_seconds," *
        "metal_compile_warm_seconds,metal_min_seconds,metal_median_seconds," *
        "speedup_median,value_relative_error,gradient_relative_l2_error",
    )

    for nsettings in BATCH_SIZES, chunksize in CHUNKSIZES
        prepared = MUQ.prepare_booster_batch(
            operating_points(nsettings),
            zeros(nsettings, MUQ.N_BPMS);
            noise_std_mm=0.2,
        )
        factors = 1 .+ 0.002 .* sin.(1:MUQ.N_QUADS)
        cpu = MUQ.BoosterBatchProblem(
            prepared;
            sensitivity=MUQ.BatchedForwardSensitivity(
                chunksize=chunksize, warm_start=true, maxiter=40
            ),
        )
        cpu_value, cpu_gradient = MUQ.value_and_gradient!(cpu, factors)
        cpu_min, cpu_median = timed_samples() do
            MUQ.value_and_gradient!(cpu, factors)
        end

        metal = metal_problem(
            prepared; chunksize=chunksize, warm_start=true, maxiter=40
        )
        warm_result = nothing
        compile_warm = @elapsed begin
            warm_result = metal_value_and_gradient!(metal, factors)
        end
        metal_value, metal_gradient = warm_result
        metal_min, metal_median = timed_samples() do
            metal_value_and_gradient!(metal, factors)
            Metal.synchronize()
        end

        value_error = abs(metal_value - cpu_value) / max(abs(cpu_value), 1.0)
        gradient_error = norm(metal_gradient - cpu_gradient) /
            max(norm(cpu_gradient), eps())
        speedup = cpu_median / metal_median
        @printf(
            "batch=%d chunk=%d cpu=%.6f metal=%.6f speedup=%.3f value_error=%.3e gradient_error=%.3e warm=%.3f\n",
            nsettings,
            chunksize,
            cpu_median,
            metal_median,
            speedup,
            value_error,
            gradient_error,
            compile_warm,
        )
        @printf(
            io,
            "%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9e,%.9e\n",
            nsettings,
            chunksize,
            cpu_min,
            cpu_median,
            compile_warm,
            metal_min,
            metal_median,
            speedup,
            value_error,
            gradient_error,
        )
        flush(io)
    end
end

println("wrote $OUTPUT")
