module BoosterBatchedUQCUDA

using ChainRulesCore
using CUDA
using ForwardDiff
using KernelAbstractions
using SciBmad

include(joinpath(@__DIR__, "..", "booster_batched_uq.jl"))
using .BoosterBatchedUQ
import .BoosterBatchedUQ: value_and_gradient!, factor_batch_value_and_gradient!,
    booster_loglikelihood, booster_factor_batch_loglikelihoods, batch_problem,
    _with_workspace, _prepare_factors!, _solve_orbit!, _likelihood!, _gradient!

const UQ = BoosterBatchedUQ
include(joinpath(@__DIR__, "..", "booster_gpu_common.jl"))
const _CUDA_KERNEL_ADAPTOR = isdefined(CUDA, :KernelAdaptor) ?
    CUDA.KernelAdaptor : CUDA.Adaptor

# Compatibility bridges for lane-dependent BeamTracking parameters.  Current
# CUDA.jl calls Adapt through `CUDA.KernelAdaptor`; the fallback supports older
# CUDA.jl releases that named this type `CUDA.Adaptor`.
function UQ.BeamTracking.Adapt.adapt_structure(
    to::_CUDA_KERNEL_ADAPTOR,
    parameter::UQ.BeamTracking._LoweredBatchParam{N},
) where {N}
    batch = UQ.BeamTracking.Adapt.adapt(to, parameter.batch)
    UQ.BeamTracking._LoweredBatchParam{N}(batch)
end

function UQ.BeamTracking.Adapt.adapt_structure(
    to::_CUDA_KERNEL_ADAPTOR,
    reference::UQ.BeamTracking.RefState{R},
) where {R}
    adapt = value -> UQ.BeamTracking.Adapt.adapt(to, value)
    UQ.BeamTracking.RefState{R}(
        adapt(reference.t_enter),
        adapt(reference.beta_gamma_enter),
        adapt(reference.t_exit),
        adapt(reference.beta_gamma_exit),
        adapt(reference.L),
        adapt(reference.g),
        adapt(reference.ds_step),
    )
end

export CUDABoosterBatchProblem,
    CUDABoosterFactorBatchProblem,
    batch_problem,
    booster_loglikelihood,
    booster_factor_batch_loglikelihoods,
    factor_batch_value_and_gradient!,
    device_status,
    loglikelihood_value!,
    value_and_gradient!,
    cuda_booster_loglikelihood,
    cuda_factor_batch_loglikelihoods,
    cuda_factor_loglikelihood_values!,
    cuda_factor_problem,
    cuda_factor_value_and_gradient!,
    cuda_loglikelihood_value!,
    cuda_problem,
    cuda_value_and_gradient!,
    cuda_chain_problems

"""Allocate one independent CUDA workspace per inference chain."""
function cuda_chain_problems(prepared; nchains::Int=Threads.nthreads(), kwargs...)
    nchains > 0 || throw(ArgumentError("nchains must be positive"))
    [cuda_problem(prepared; kwargs...) for _ in 1:nchains]
end

"""Device-resident closed-orbit and tracking workspace for CUDA."""
mutable struct CUDABoosterBatchProblem{O,D,LDM,FM,FHM,FDM,VDM,GDM,IM,TM}
    orbit::O
    device::D
    lhs_work::LDM
    failed::FM
    factors_host::FHM
    factors_device::FDM
    values_device::VDM
    gradient_device::GDM
    input_invalid::IM
    tracking_lost::TM
    nexperiments::Int
    nfactors::Int
    busy::Bool
end

"""CUDA workspace with independent factor and experiment axes."""
mutable struct CUDABoosterFactorBatchProblem{O,D,LDM,FM,FHM,FDM,VDM,GDM,IM,TM}
    orbit::O
    device::D
    lhs_work::LDM
    failed::FM
    factors_host::FHM
    factors_device::FDM
    values_device::VDM
    gradient_device::GDM
    input_invalid::IM
    tracking_lost::TM
    nexperiments::Int
    nfactors::Int
    busy::Bool
end

_cuda_array(values, ::Type{T}) where {T<:AbstractFloat} = CuArray(T.(values))

function _cuda_prepared(prepared::UQ.PreparedBoosterBatch, ::Type{T}) where {T}
    UQ.PreparedBoosterBatch(
        _cuda_array(prepared.p_over_q_ref, T),
        _cuda_array(prepared.dipole_k0, T),
        _cuda_array(prepared.dipole_k1, T),
        _cuda_array(prepared.dipole_k2, T),
        _cuda_array(prepared.quad_k1_base, T),
        _cuda_array(prepared.sext_k2, T),
        _cuda_array(prepared.hcorrector_k0L, T),
        _cuda_array(prepared.vcorrector_k0L, T),
        _cuda_array(prepared.ac_quad_k1, T),
        _cuda_array(prepared.injection_k0L, T),
        _cuda_array(prepared.extraction_f3_k0L, T),
        _cuda_array(prepared.extraction_d3_k0L, T),
        _cuda_array(prepared.observed_orbit_mm, T),
        _cuda_array(prepared.inv_variance_mm, T),
        T(prepared.loglik_constant),
    )
end

"""
    cuda_problem(prepared; chunksize=4, device_eltype=Float64, ...)

Prepare persistent CUDA lattices and work buffers. Transfer-function preparation
runs once on the host. The closed-orbit solve, primal/tangent tracking, implicit
sensitivities, and likelihood/gradient reductions run on the GPU.
Use one problem per concurrently executing inference chain.
"""
function cuda_problem(
    prepared::UQ.PreparedBoosterBatch;
    chunksize::Int=4,
    device_eltype::Type{T}=Float64,
    warm_start::Bool=true,
    abstol::Real=1e-12,
    maxiter::Int=30,
) where {T<:AbstractFloat}
    CUDA.functional() || error("CUDA.jl reports that no functional CUDA device is available")
    T in (Float32, Float64) || throw(ArgumentError(
        "device_eltype must be Float32 or Float64, received $T"
    ))
    1 <= chunksize <= UQ.N_QUADS || throw(ArgumentError(
        "chunksize must lie in 1:$(UQ.N_QUADS), received $chunksize"
    ))

    device_prepared = _cuda_prepared(prepared, T)
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
    failed = CuArray(fill(false, nsettings))
    factors_host = zeros(T, 1, UQ.N_QUADS)
    CUDABoosterBatchProblem(
        _orbit_workspace(device), device, lhs_work, failed,
        factors_host, CuArray(factors_host), CuArray(zeros(T, 1)),
        CuArray(zeros(T, 1, UQ.N_QUADS)), CuArray(fill(false, 1)),
        CuArray(fill(false, nsettings)),
        nsettings, 1, false,
    )
end

"""
    cuda_factor_problem(prepared, nfactors; chunksize=4, device_eltype=Float64, ...)

Prepare a fixed-size CUDA workspace for `nfactors` independent quadrupole
scaling vectors evaluated against the same Booster experiments.  The device
batch is the flattened Cartesian product `(factor, experiment)`, with the
experiment index varying fastest. Orbit solves, tracking, implicit sensitivities,
and reductions use `device_eltype` on CUDA.

Width 4 is the recommended CUDA chunk size.  Larger dual widths can consume
prohibitive per-thread local memory in BeamTracking kernels, particularly in
Float64, and should be treated as device-specific experiments.
"""
function cuda_factor_problem(
    prepared::UQ.PreparedBoosterBatch,
    nfactors::Integer;
    chunksize::Int=4,
    device_eltype::Type{T}=Float64,
    warm_start::Bool=true,
    abstol::Real=1e-12,
    maxiter::Int=30,
) where {T<:AbstractFloat}
    CUDA.functional() || error("CUDA.jl reports that no functional CUDA device is available")
    nfactors > 0 || throw(ArgumentError("nfactors must be positive"))
    T in (Float32, Float64) || throw(ArgumentError(
        "device_eltype must be Float32 or Float64, received $T"
    ))
    1 <= chunksize <= UQ.N_QUADS || throw(ArgumentError(
        "chunksize must lie in 1:$(UQ.N_QUADS), received $chunksize"
    ))

    device_prepared = _cuda_prepared(
        UQ._repeat_prepared_for_factors(prepared, Int(nfactors)), T
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
    factors_host = zeros(T, nfactors_int, UQ.N_QUADS)
    CUDABoosterFactorBatchProblem(
        _orbit_workspace(device),
        device,
        similar(device.primal.fixed_point_lhs),
        CuArray(fill(false, ncombined)),
        factors_host,
        CuArray(factors_host),
        CuArray(zeros(T, nfactors_int)),
        CuArray(zeros(T, nfactors_int, UQ.N_QUADS)),
        CuArray(fill(false, nfactors_int)),
        CuArray(fill(false, ncombined)),
        length(prepared),
        nfactors_int,
        false,
    )
end

batch_problem(prepared::UQ.PreparedBoosterBatch, ::CUDA.CUDABackend; kwargs...) =
    cuda_problem(prepared; kwargs...)
batch_problem(
    prepared::UQ.PreparedBoosterBatch, nfactors::Integer,
    ::CUDA.CUDABackend; kwargs...,
) = cuda_factor_problem(prepared, nfactors; kwargs...)

@inline _partial(value, ::Val{I}) where {I} = ForwardDiff.partials(value, I)

@generated function _zero_dual(
    ::Type{D}, value
) where {D<:ForwardDiff.Dual}
    T = ForwardDiff.valtype(D)
    C = ForwardDiff.npartials(D)
    partials = Expr(:tuple, [:(zero($T)) for _ in 1:C]...)
    :($D(value, ForwardDiff.Partials($partials)))
end

const _CUDAProblem = Union{CUDABoosterBatchProblem,CUDABoosterFactorBatchProblem}

function _with_workspace(f, problem::_CUDAProblem)
    problem.busy && error("$(typeof(problem)) is already in use; create one problem per chain")
    problem.busy = true
    try
        result = f()
        CUDA.synchronize()
        result
    finally
        problem.busy = false
    end
end

function _stage_factors!(problem::_CUDAProblem, factors::CuArray)
    eltype(factors) === eltype(problem.factors_device) || throw(ArgumentError(
        "device factors must use $(eltype(problem.factors_device))"
    ))
    copyto!(vec(problem.factors_device), vec(factors))
end

function _stage_factors!(problem::_CUDAProblem, factors::AbstractArray)
    all(isfinite, factors) || throw(ArgumentError("quad_factors must be finite"))
    vec(problem.factors_host) .= vec(factors)
    copyto!(problem.factors_device, problem.factors_host)
end

function _copy_factor_values!(problem::CUDABoosterBatchProblem, factors::AbstractVector)
    length(factors) == UQ.N_QUADS || throw(DimensionMismatch(
        "quad_factors must contain $(UQ.N_QUADS) values"
    ))
    _stage_factors!(problem, factors)
end

function _copy_factor_values!(problem::CUDABoosterFactorBatchProblem, factors::AbstractMatrix)
    size(factors) == (problem.nfactors, UQ.N_QUADS) || throw(DimensionMismatch(
        "quad_factors must have size ($(problem.nfactors), $(UQ.N_QUADS))"
    ))
    _stage_factors!(problem, factors)
end

@kernel function _validate_factors_kernel!(invalid, factors)
    factor = @index(Global, Linear)
    bad = false
    @inbounds for parameter in axes(factors, 2)
        bad |= !isfinite(factors[factor, parameter])
    end
    invalid[factor] = bad
end

@kernel function _set_active_quadrupoles_kernel!(active, base, factors, nexperiments)
    row, parameter = @index(Global, NTuple)
    factor = (row - 1) ÷ nexperiments + 1
    @inbounds active[row, parameter] = base[row, parameter] * factors[factor, parameter]
end

function _update_device_primal!(problem::_CUDAProblem)
    backend = KernelAbstractions.get_backend(problem.factors_device)
    _validate_factors_kernel!(backend)(
        problem.input_invalid, problem.factors_device; ndrange=problem.nfactors,
    )
    active = problem.device.primal.quad_active
    _set_active_quadrupoles_kernel!(backend)(
        active, problem.device.prepared.quad_k1_base, problem.factors_device,
        problem.nexperiments; ndrange=size(active),
    )
    nothing
end

function _update_device_primal!(problem::CUDABoosterBatchProblem, factors::AbstractVector)
    _copy_factor_values!(problem, factors)
    _update_device_primal!(problem)
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
    values[factor] = loglik_constant / T(size(values, 1)) - T(0.5) * total
end

function _device_prediction_and_weights!(problem::_CUDAProblem)
    device = problem.device
    primal = device.primal
    prepared = device.prepared
    T = eltype(primal.tracking_state)
    copyto!(primal.tracking_state, primal.closed_orbit)
    UQ._track!(
        primal.tracking_state, primal.lattice;
        samples=primal.predicted_orbit_mm, threaded=false,
        lost=problem.tracking_lost,
    )
    primal.predicted_orbit_mm .*= T(1_000)
    backend = KernelAbstractions.get_backend(problem.values_device)
    _factor_likelihood_kernel!(backend)(
        problem.values_device, primal.residual_weight,
        primal.predicted_orbit_mm, prepared.observed_orbit_mm,
        prepared.inv_variance_mm, problem.nexperiments,
        prepared.loglik_constant;
        ndrange=problem.nfactors,
    )
    problem.values_device
end

function _prepare_factors!(problem::_CUDAProblem, factors)
    fill!(problem.failed, false)
    fill!(problem.tracking_lost, false)
    _copy_factor_values!(problem, factors)
    _update_device_primal!(problem)
end

function _gradient!(problem::_CUDAProblem, factors)
    fill!(problem.gradient_device, zero(eltype(problem.gradient_device)))
    fill!(problem.failed, false)
    for first_parameter in 1:problem.device.sensitivity.chunksize:UQ.N_QUADS
        _device_factor_gradient_chunk!(problem, first_parameter, Val(true))
    end
    problem.gradient_device
end

"""Device status buffers; values are valid until the next evaluation on this workspace."""
device_status(problem::_CUDAProblem) = (;
    converged=problem.orbit.converged,
    orbit_failed=problem.orbit.failed,
    sensitivity_failed=problem.failed,
    input_invalid=problem.input_invalid,
    tracking_lost=problem.tracking_lost,
)

include(joinpath(@__DIR__, "booster_scibmad_cuda.jl"))
_likelihood!(problem::_CUDAProblem) = _device_prediction_and_weights!(problem)
_evaluate_device!(problem::_CUDAProblem, factors, mode) =
    UQ._evaluate_core!(problem, factors, mode)

function _check_device_status!(problem::_CUDAProblem)
    status = device_status(problem)
    any(Array(status.input_invalid)) && error("non-finite CUDA quadrupole factor")
    any(Array(status.tracking_lost)) && error("a CUDA particle was lost during tracking")
    any(Array(status.orbit_failed)) && error("singular CUDA closed-orbit Jacobian")
    all(Array(status.converged)) || error("CUDA closed orbit did not converge")
    any(Array(status.sensitivity_failed)) && error("singular CUDA implicit sensitivity")
    nothing
end

_host_result(::CUDABoosterBatchProblem, values::CuArray) = Float64(Array(values)[1])
_host_result(::CUDABoosterFactorBatchProblem, values::CuArray) = Float64.(Array(values))
_host_result(problem::CUDABoosterBatchProblem, result::Tuple) =
    (_host_result(problem, result[1]), Float64.(vec(Array(result[2]))))
_host_result(problem::CUDABoosterFactorBatchProblem, result::Tuple) =
    (_host_result(problem, result[1]), Float64.(Array(result[2])))

function _evaluate_host!(problem::_CUDAProblem, factors, mode)
    result = _evaluate_device!(problem, factors, mode)
    _check_device_status!(problem)
    _host_result(problem, result)
end

# Device factor inputs retain device values/gradients; host inputs are explicit
# entry/exit adapters for existing inference callers.
value_and_gradient!(problem::CUDABoosterBatchProblem, factors::CuArray{<:Real,1}) =
    _evaluate_device!(problem, factors, Val(true))
value_and_gradient!(problem::CUDABoosterBatchProblem, factors::AbstractVector{<:Real}) =
    _evaluate_host!(problem, factors, Val(true))
factor_batch_value_and_gradient!(
    problem::CUDABoosterFactorBatchProblem, factors::CuArray{<:Real,2}
) = _evaluate_device!(problem, factors, Val(true))
factor_batch_value_and_gradient!(
    problem::CUDABoosterFactorBatchProblem, factors::AbstractMatrix{<:Real}
) = _evaluate_host!(problem, factors, Val(true))
value_and_gradient!(
    problem::CUDABoosterFactorBatchProblem, factors::AbstractMatrix{<:Real}
) = factor_batch_value_and_gradient!(problem, factors)

loglikelihood_value!(problem::CUDABoosterBatchProblem, factors::CuArray{<:Real,1}) =
    _evaluate_device!(problem, factors, Val(false))
loglikelihood_value!(problem::CUDABoosterBatchProblem, factors::AbstractVector{<:Real}) =
    _evaluate_host!(problem, factors, Val(false))
loglikelihood_value!(problem::CUDABoosterFactorBatchProblem, factors::CuArray{<:Real,2}) =
    _evaluate_device!(problem, factors, Val(false))
loglikelihood_value!(problem::CUDABoosterFactorBatchProblem, factors::AbstractMatrix{<:Real}) =
    _evaluate_host!(problem, factors, Val(false))

booster_loglikelihood(problem::CUDABoosterBatchProblem, factors::AbstractVector{<:Real}) =
    first(value_and_gradient!(problem, factors))
booster_factor_batch_loglikelihoods(
    problem::CUDABoosterFactorBatchProblem, factors::AbstractMatrix{<:Real}
) = first(factor_batch_value_and_gradient!(problem, factors))

function ChainRulesCore.rrule(
    ::typeof(booster_factor_batch_loglikelihoods),
    problem::CUDABoosterFactorBatchProblem,
    factors::AbstractMatrix{<:Real},
)
    values, gradient = factor_batch_value_and_gradient!(problem, factors)
    function pullback(tangent)
        scale = unthunk(tangent)
        factor_tangent = scale isa AbstractZero ?
            ZeroTangent() : ProjectTo(factors)(reshape(scale, :, 1) .* gradient)
        NoTangent(), NoTangent(), factor_tangent
    end
    values, pullback
end

function ChainRulesCore.rrule(
    ::typeof(booster_loglikelihood),
    problem::CUDABoosterBatchProblem,
    factors::AbstractVector{<:Real},
)
    value, gradient = value_and_gradient!(problem, factors)
    function pullback(tangent)
        scale = unthunk(tangent)
        factor_tangent = scale isa AbstractZero ?
            ZeroTangent() : ProjectTo(factors)(scale .* gradient)
        NoTangent(), NoTangent(), factor_tangent
    end
    value, pullback
end

function ChainRulesCore.rrule(
    ::typeof(booster_loglikelihood),
    problem::CUDABoosterBatchProblem,
    factors::CuArray{<:Real,1},
)
    values, gradient = value_and_gradient!(problem, factors)
    function pullback(tangent)
        scale = unthunk(tangent)
        factor_tangent = scale isa AbstractZero ? ZeroTangent() :
            ProjectTo(factors)(vec(gradient .* (
                scale isa AbstractVector ? reshape(scale, 1, 1) : scale
            )))
        NoTangent(), NoTangent(), factor_tangent
    end
    values, pullback
end

# Compatibility names for existing callers; all route through the dispatched API.
const cuda_value_and_gradient! = value_and_gradient!
const cuda_factor_value_and_gradient! = factor_batch_value_and_gradient!
const cuda_loglikelihood_value! = loglikelihood_value!
const cuda_factor_loglikelihood_values! = loglikelihood_value!
const cuda_booster_loglikelihood = booster_loglikelihood
const cuda_factor_batch_loglikelihoods = booster_factor_batch_loglikelihoods

end
