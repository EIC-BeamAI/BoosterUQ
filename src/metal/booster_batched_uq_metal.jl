module BoosterBatchedUQMetal

using ChainRulesCore
using ForwardDiff
using KernelAbstractions
using Metal

include(joinpath(@__DIR__, "..", "booster_batched_uq.jl"))
using .BoosterBatchedUQ

const UQ = BoosterBatchedUQ
include(joinpath(@__DIR__, "..", "booster_gpu_common.jl"))
include(joinpath(@__DIR__, "booster_metal_device.jl"))

export MetalBoosterBatchProblem,
    MetalBoosterFactorBatchProblem,
    metal_booster_loglikelihood,
    metal_factor_batch_loglikelihoods,
    metal_factor_problem,
    metal_factor_value_and_gradient!,
    metal_problem,
    metal_value_and_gradient!,
    metal_chain_problems

"""Allocate one independent Metal workspace per inference chain."""
function metal_chain_problems(prepared; nchains::Int=Threads.nthreads(), kwargs...)
    nchains > 0 || throw(ArgumentError("nchains must be positive"))
    [metal_problem(prepared; kwargs...) for _ in 1:nchains]
end

"""Device-resident closed-orbit and tracking workspace for Apple Metal."""
mutable struct MetalBoosterBatchProblem{O,D,LDM,FM}
    orbit::O
    device::D
    lhs_work::LDM
    failed::FM
    busy::Bool
end

"""Metal workspace with independent factor and experiment axes."""
mutable struct MetalBoosterFactorBatchProblem{O,D,LDM,FM,FHM,FDM,VDM,GDM}
    orbit::O
    device::D
    lhs_work::LDM
    failed::FM
    factors_host32::FHM
    factors_device::FDM
    values_device::VDM
    gradient_device::GDM
    nexperiments::Int
    nfactors::Int
    busy::Bool
end

_metal_array(values) = MtlArray(Float32.(values))

function _metal_prepared(prepared::UQ.PreparedBoosterBatch)
    UQ.PreparedBoosterBatch(
        _metal_array(prepared.p_over_q_ref),
        _metal_array(prepared.dipole_k0),
        _metal_array(prepared.dipole_k1),
        _metal_array(prepared.dipole_k2),
        _metal_array(prepared.quad_k1_base),
        _metal_array(prepared.sext_k2),
        _metal_array(prepared.hcorrector_k0L),
        _metal_array(prepared.vcorrector_k0L),
        _metal_array(prepared.ac_quad_k1),
        _metal_array(prepared.injection_k0L),
        _metal_array(prepared.extraction_f3_k0L),
        _metal_array(prepared.extraction_d3_k0L),
        _metal_array(prepared.observed_orbit_mm),
        _metal_array(prepared.inv_variance_mm),
        Float32(prepared.loglik_constant),
    )
end

"""
    metal_problem(prepared; chunksize=8, warm_start=true, ...)

Prepare persistent Float32 Metal lattices and buffers. Transfer-function
preparation runs once on the host; the closed-orbit solve, BPM tracking,
implicit sensitivities, and likelihood reductions run on the Apple GPU.
"""
function metal_problem(
    prepared::UQ.PreparedBoosterBatch;
    chunksize::Int=8,
    warm_start::Bool=true,
    abstol::Real=1e-12,
    maxiter::Int=30,
)
    Metal.functional() || error("Metal.jl reports that the Apple GPU is unavailable")
    device_prepared = _metal_prepared(prepared)
    device = UQ.BoosterBatchProblem(
        device_prepared;
        sensitivity=UQ.BatchedForwardSensitivity(
            chunksize=chunksize,
            warm_start=warm_start,
            abstol=Float64(abstol),
            maxiter=maxiter,
        ),
    )
    nsettings = length(prepared)
    lhs_work = similar(device.primal.fixed_point_lhs)
    failed = Metal.MtlArray(fill(false, nsettings))
    MetalBoosterBatchProblem(
        _orbit_workspace(device), device, lhs_work, failed, false,
    )
end

"""
    metal_factor_problem(prepared, nfactors; chunksize=8, ...)

Prepare a fixed-size Metal workspace for `nfactors` independent quadrupole
scaling vectors evaluated against the same `prepared` Booster experiments.
Orbit solves and tracking both use the flattened Cartesian product on device,
while likelihoods and gradients retain their factor-set axis.
"""
function metal_factor_problem(
    prepared::UQ.PreparedBoosterBatch,
    nfactors::Integer;
    chunksize::Int=8,
    warm_start::Bool=true,
    abstol::Real=1e-12,
    maxiter::Int=30,
)
    Metal.functional() || error("Metal.jl reports that the Apple GPU is unavailable")
    nfactors > 0 || throw(ArgumentError("nfactors must be positive"))
    1 <= chunksize <= UQ.N_QUADS || throw(ArgumentError(
        "chunksize must lie in 1:$(UQ.N_QUADS), received $chunksize"
    ))
    device_prepared = _metal_prepared(
        UQ._repeat_prepared_for_factors(prepared, Int(nfactors))
    )
    device = UQ.BoosterBatchProblem(
        device_prepared;
        sensitivity=UQ.BatchedForwardSensitivity(
            chunksize=chunksize,
            warm_start=warm_start,
            abstol=Float64(abstol),
            maxiter=maxiter,
        ),
    )
    ncombined = length(device_prepared)
    nfactors_int = Int(nfactors)
    factors_host32 = zeros(Float32, nfactors_int, UQ.N_QUADS)
    MetalBoosterFactorBatchProblem(
        _orbit_workspace(device),
        device,
        similar(device.primal.fixed_point_lhs),
        Metal.MtlArray(fill(false, ncombined)),
        factors_host32,
        Metal.MtlArray(factors_host32),
        Metal.MtlArray(zeros(Float32, nfactors_int)),
        Metal.MtlArray(zeros(Float32, nfactors_int, UQ.N_QUADS)),
        length(prepared),
        nfactors_int,
        false,
    )
end

@inline _partial(value, ::Val{I}) where {I} = ForwardDiff.partials(value, I)

@generated function _zero_dual(
    ::Type{D}, value
) where {D<:ForwardDiff.Dual}
    T = ForwardDiff.valtype(D)
    C = ForwardDiff.npartials(D)
    partials = Expr(:tuple, [:(zero($T)) for _ in 1:C]...)
    :($D(value, ForwardDiff.Partials($partials)))
end

function _update_device_primal!(problem::MetalBoosterBatchProblem, factors)
    device = problem.device
    T = eltype(device.prepared.quad_k1_base)
    UQ._update_quadrupoles!(
        device.primal.quad_active,
        device.prepared.quad_k1_base,
        T.(factors),
    )
    nothing
end

function _copy_factor_values!(problem::MetalBoosterFactorBatchProblem, factors)
    size(factors) == (problem.nfactors, UQ.N_QUADS) ||
        throw(DimensionMismatch(
            "quad_factors must have size ($(problem.nfactors), $(UQ.N_QUADS))"
        ))
    all(isfinite, factors) || throw(ArgumentError("quad_factors must be finite"))
    problem.factors_host32 .= factors
    copyto!(problem.factors_device, problem.factors_host32)
    nothing
end

function _update_device_primal!(problem::MetalBoosterFactorBatchProblem)
    device = problem.device
    UQ._update_factor_batch_quadrupoles!(
        device.primal.quad_active,
        device.prepared.quad_k1_base,
        problem.factors_device,
        problem.nexperiments,
    )
    nothing
end

function _device_prediction_and_weights!(problem::MetalBoosterBatchProblem)
    device = problem.device
    primal = device.primal
    prepared = device.prepared
    primal.tracking_state .= primal.closed_orbit
    UQ._track!(
        primal.tracking_state,
        primal.lattice;
        samples=primal.predicted_orbit_mm,
        threaded=false,
    )
    primal.predicted_orbit_mm .*= 1_000f0
    primal.residual_weight .=
        (prepared.observed_orbit_mm .- primal.predicted_orbit_mm) .*
        prepared.inv_variance_mm
    weighted_sum_of_squares = sum(
        (prepared.observed_orbit_mm .- primal.predicted_orbit_mm) .*
        primal.residual_weight
    )
    Float64(prepared.loglik_constant - 0.5f0weighted_sum_of_squares)
end

@kernel function _factor_likelihood_kernel!(
    values,
    residual_weight,
    predicted,
    observed,
    inv_variance,
    nexperiments,
    loglik_constant,
)
    factor = @index(Global, Linear)
    T = eltype(values)
    total = zero(T)
    first_row = nexperiments * (factor - 1) + 1
    last_row = first_row + nexperiments - 1
    for bpm in axes(predicted, 2), row in first_row:last_row
        residual = observed[row, bpm] - predicted[row, bpm]
        weight = residual * inv_variance[row, bpm]
        residual_weight[row, bpm] = weight
        total += residual * weight
    end
    values[factor] = loglik_constant - T(0.5) * total
end

function _device_prediction_and_weights!(problem::MetalBoosterFactorBatchProblem)
    device = problem.device
    primal = device.primal
    prepared = device.prepared
    primal.tracking_state .= primal.closed_orbit
    UQ._track!(
        primal.tracking_state,
        primal.lattice;
        samples=primal.predicted_orbit_mm,
        threaded=false,
    )
    primal.predicted_orbit_mm .*= 1_000f0
    backend = KernelAbstractions.get_backend(problem.values_device)
    _factor_likelihood_kernel!(backend)(
        problem.values_device,
        primal.residual_weight,
        primal.predicted_orbit_mm,
        prepared.observed_orbit_mm,
        prepared.inv_variance_mm,
        problem.nexperiments,
        Float32(device.prepared.loglik_constant / problem.nfactors);
        ndrange=problem.nfactors,
    )
    KernelAbstractions.synchronize(backend)
    Float64.(Array(problem.values_device))
end

function _with_workspace(f, problem::MetalBoosterBatchProblem)
    problem.busy && error(
        "MetalBoosterBatchProblem is already in use; create one problem per chain"
    )
    problem.busy = true
    try
        f()
    finally
        problem.busy = false
    end
end

function _with_workspace(f, problem::MetalBoosterFactorBatchProblem)
    problem.busy && error(
        "MetalBoosterFactorBatchProblem is already in use; create one problem per chain"
    )
    problem.busy = true
    try
        f()
    finally
        problem.busy = false
    end
end

"""Evaluate the Booster likelihood and shared quadrupole gradient using Metal."""
function metal_value_and_gradient!(
    problem::MetalBoosterBatchProblem,
    quad_factors::AbstractVector{<:Real},
)
    _with_workspace(problem) do
        length(quad_factors) == UQ.N_QUADS || throw(DimensionMismatch(
            "quad_factors must contain $(UQ.N_QUADS) values"
        ))
        all(isfinite, quad_factors) ||
            throw(ArgumentError("quad_factors must be finite"))

        _update_device_primal!(problem, quad_factors)
        _solve_device_orbit!(problem)
        loglikelihood = _device_prediction_and_weights!(problem)
        gradient = zeros(Float64, UQ.N_QUADS)
        chunksize = problem.device.sensitivity.chunksize
        for first_parameter in 1:chunksize:UQ.N_QUADS
            _device_gradient_chunk!(
                gradient, problem, quad_factors, first_parameter
            )
        end
        Metal.synchronize()
        loglikelihood, gradient
    end
end

"""Evaluate independent Metal likelihoods and gradients for factor-matrix rows."""
function metal_factor_value_and_gradient!(
    problem::MetalBoosterFactorBatchProblem,
    quad_factors::AbstractMatrix{<:Real},
)
    _with_workspace(problem) do
        _copy_factor_values!(problem, quad_factors)
        _update_device_primal!(problem)
        _solve_device_orbit!(problem)
        loglikelihoods = _device_prediction_and_weights!(problem)
        fill!(problem.gradient_device, 0.0f0)
        chunksize = problem.device.sensitivity.chunksize
        for first_parameter in 1:chunksize:UQ.N_QUADS
            _device_factor_gradient_chunk!(problem, first_parameter)
        end
        Metal.synchronize()
        loglikelihoods, Float64.(Array(problem.gradient_device))
    end
end

metal_factor_batch_loglikelihoods(
    problem::MetalBoosterFactorBatchProblem,
    quad_factors::AbstractMatrix{<:Real},
) = first(metal_factor_value_and_gradient!(problem, quad_factors))

function ChainRulesCore.rrule(
    ::typeof(metal_factor_batch_loglikelihoods),
    problem::MetalBoosterFactorBatchProblem,
    quad_factors::AbstractMatrix{<:Real},
)
    loglikelihoods, gradient = metal_factor_value_and_gradient!(
        problem, quad_factors
    )
    function pullback(loglikelihood_tangent)
        scale = unthunk(loglikelihood_tangent)
        factor_tangent = scale isa AbstractZero ?
            ZeroTangent() :
            ProjectTo(quad_factors)(reshape(scale, :, 1) .* gradient)
        NoTangent(), NoTangent(), factor_tangent
    end
    loglikelihoods, pullback
end

metal_booster_loglikelihood(problem::MetalBoosterBatchProblem, quad_factors) =
    first(metal_value_and_gradient!(problem, quad_factors))

function ChainRulesCore.rrule(
    ::typeof(metal_booster_loglikelihood),
    problem::MetalBoosterBatchProblem,
    quad_factors::AbstractVector{<:Real},
)
    loglikelihood, gradient = metal_value_and_gradient!(problem, quad_factors)
    function pullback(loglikelihood_tangent)
        NoTangent(), NoTangent(), loglikelihood_tangent .* gradient
    end
    loglikelihood, pullback
end

end
