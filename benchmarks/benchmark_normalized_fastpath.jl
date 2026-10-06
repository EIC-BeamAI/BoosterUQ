using Printf
using Statistics

include(joinpath(@__DIR__, "..", "booster_batched_uq.jl"))
using .BoosterBatchedUQ

const UQ = BoosterBatchedUQ
const NSETTINGS = parse(Int, get(ENV, "BOOSTER_FASTPATH_BATCH", "256"))
const CHUNKSIZE = parse(Int, get(ENV, "BOOSTER_FASTPATH_CHUNK", "48"))
const SAMPLES = parse(Int, get(ENV, "BOOSTER_FASTPATH_SAMPLES", "5"))

function operating_points(nsettings)
    default = UQ._BoosterLatticeTemplate.default_booster_operating_point()
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

points = operating_points(NSETTINGS)
prepared = prepare_booster_batch(
    points,
    zeros(NSETTINGS, UQ.N_BPMS);
    noise_std_mm=0.2,
)
problem = BoosterBatchProblem(
    prepared;
    sensitivity=BatchedForwardSensitivity(
        chunksize=CHUNKSIZE,
        warm_start=true,
        maxiter=40,
    ),
)
factors = ones(UQ.N_QUADS)
value_and_gradient!(problem, factors)

times = Float64[]
bytes = Int[]
for _ in 1:SAMPLES
    GC.gc()
    result = @timed value_and_gradient!(problem, factors)
    push!(times, result.time)
    push!(bytes, result.bytes)
end

@printf(
    "fastpath=%s rows=%d chunk=%d samples=%d min_seconds=%.9f median_seconds=%.9f median_bytes=%.0f\n",
    UQ.NORMALIZED_BATCH_FASTPATH_ENABLED ? "enabled" : "disabled",
    NSETTINGS,
    CHUNKSIZE,
    SAMPLES,
    minimum(times),
    median(times),
    median(bytes),
)
