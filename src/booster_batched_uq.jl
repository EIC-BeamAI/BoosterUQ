module BoosterBatchedUQ

using Beamlines
using BeamTracking
using ChainRulesCore
using ForwardDiff
using KernelAbstractions
using LinearAlgebra
using SciBmad
import ReverseDiff
import Turing

export BatchedForwardSensitivity,
    batch_problem,
    BoosterBatchProblem,
    BoosterFactorBatchProblem,
    CORRECTOR_NAMES,
    H_CORRECTOR_NAMES,
    PreparedBoosterBatch,
    V_CORRECTOR_NAMES,
    booster_loglikelihood,
    booster_factor_batch_loglikelihoods,
    chain_problems,
    factor_batch_value_and_gradient!,
    booster_model,
    lognormal_quad_prior,
    predict_orbits,
    prepare_booster_batch,
    value_and_gradient!

include(joinpath(@__DIR__, "booster_uq_compat.jl"))
include(joinpath(@__DIR__, "booster_uq_preparation.jl"))
include(joinpath(@__DIR__, "booster_uq_workspaces.jl"))
include(joinpath(@__DIR__, "booster_uq_orbit.jl"))
include(joinpath(@__DIR__, "booster_scibmad_reference.jl"))
include(joinpath(@__DIR__, "booster_uq_inference.jl"))

end # module BoosterBatchedUQ
