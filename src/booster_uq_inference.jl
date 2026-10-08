@inline function _dual(::Type{D}, value, partials::NTuple{C,T}) where {D,C,T}
    D(value, ForwardDiff.Partials(partials))
end

function _seed_quadrupoles!(jvp::_JVPWorkspace{C,D}, factors, first_parameter) where {C,D}
    T = ForwardDiff.valtype(D)
    for quad in 1:N_QUADS
        partials = ntuple(C) do local_parameter
            quad == first_parameter + local_parameter - 1 ? one(T) : zero(T)
        end
        jvp.quad_seed[quad] = _dual(D, T(factors[quad]), partials)
    end
    jvp.quad_seed
end

function _seed_fixed_orbit!(state::AbstractMatrix{D}, closed_orbit) where {D<:ForwardDiff.Dual}
    C = ForwardDiff.npartials(D)
    T = ForwardDiff.valtype(D)
    zero_partials = ntuple(_ -> zero(T), C)
    for index in eachindex(state)
        state[index] = _dual(D, closed_orbit[index], zero_partials)
    end
    state
end

function _extract_fixed_orbit_response!(jvp::_JVPWorkspace{C}, problem) where {C}
    for setting in axes(jvp.tracking_state, 1), coordinate in 1:4
        value = jvp.tracking_state[setting, coordinate]
        for direction in 1:C
            jvp.closed_orbit_response[setting, coordinate, direction] =
                ForwardDiff.partials(value, direction)
        end
    end
    for setting in axes(jvp.closed_orbit_response, 1)
        lhs = Matrix(@view problem.primal.fixed_point_lhs[setting, :, :])
        rhs = Matrix(@view jvp.closed_orbit_response[setting, :, :])
        response = try
            lhs \ rhs
        catch exception
            throw(ErrorException(
                "closed-orbit sensitivity solve failed for setting $setting: " *
                sprint(showerror, exception)
            ))
        end
        all(isfinite, response) || error(
            "closed-orbit sensitivity is singular or non-finite for setting $setting"
        )
        @views jvp.closed_orbit_response[setting, :, :] .= response
    end
    jvp.closed_orbit_response
end

function _seed_moving_orbit!(jvp::_JVPWorkspace{C,D}, closed_orbit) where {C,D}
    T = ForwardDiff.valtype(D)
    zero_partials = ntuple(_ -> zero(T), C)
    for setting in axes(jvp.tracking_state, 1), coordinate in 1:6
        partials = coordinate <= 4 ?
            ntuple(direction ->
                jvp.closed_orbit_response[setting, coordinate, direction], C) :
            zero_partials
        jvp.tracking_state[setting, coordinate] =
            _dual(D, closed_orbit[setting, coordinate], partials)
    end
    jvp.tracking_state
end

function _accumulate_gradient_chunk!(
    gradient,
    problem,
    quad_factors,
    first_parameter,
)
    prepared = problem.prepared
    primal = problem.primal
    jvp = problem.jvp
    C = size(jvp.closed_orbit_response, 3)
    _seed_quadrupoles!(jvp, quad_factors, first_parameter)
    _update_quadrupoles!(jvp.quad_active, prepared.quad_k1_base, jvp.quad_seed)

    # First tangent pass: dM/dq at the fixed primal closed orbit.
    _seed_fixed_orbit!(jvp.tracking_state, primal.closed_orbit)
    _track!(jvp.tracking_state, jvp.lattice; threaded=problem.sensitivity.threaded)
    _extract_fixed_orbit_response!(jvp, problem)

    # Second tangent pass: direct magnet response plus the displaced closed orbit.
    _seed_moving_orbit!(jvp, primal.closed_orbit)
    _track!(
        jvp.tracking_state,
        jvp.lattice;
        samples=jvp.sampled_orbit,
        threaded=problem.sensitivity.threaded,
    )
    for direction in 1:C
        parameter = first_parameter + direction - 1
        parameter > N_QUADS && break
        total = 0.0
        for index in eachindex(jvp.sampled_orbit)
            tangent_mm = 1_000.0 * ForwardDiff.partials(
                jvp.sampled_orbit[index], direction
            )
            total += primal.residual_weight[index] * tangent_mm
        end
        gradient[parameter] = total
    end
    gradient
end

function _seed_factor_batch_quadrupoles!(
    problem::BoosterFactorBatchProblem,
    factors,
    first_parameter,
)
    workspace = problem.workspace
    jvp = workspace.jvp
    base = workspace.prepared.quad_k1_base
    D = eltype(jvp.quad_active)
    T = ForwardDiff.valtype(D)
    C = ForwardDiff.npartials(D)
    for factor in 1:problem.nfactors, quad in 1:N_QUADS
        factor_value = T(factors[factor, quad])
        for experiment in 1:problem.nexperiments
            row = experiment + problem.nexperiments * (factor - 1)
            base_value = T(base[row, quad])
            partials = ntuple(C) do direction
                quad == first_parameter + direction - 1 ? base_value : zero(T)
            end
            jvp.quad_active[row, quad] = _dual(
                D, base_value * factor_value, partials
            )
        end
    end
    jvp.quad_active
end

function _accumulate_factor_gradient_chunk!(
    gradient,
    problem::BoosterFactorBatchProblem,
    factors,
    first_parameter,
)
    workspace = problem.workspace
    primal = workspace.primal
    jvp = workspace.jvp
    C = size(jvp.closed_orbit_response, 3)
    _seed_factor_batch_quadrupoles!(problem, factors, first_parameter)

    _seed_fixed_orbit!(jvp.tracking_state, primal.closed_orbit)
    _track!(jvp.tracking_state, jvp.lattice; threaded=workspace.sensitivity.threaded)
    _extract_fixed_orbit_response!(jvp, workspace)

    _seed_moving_orbit!(jvp, primal.closed_orbit)
    _track!(
        jvp.tracking_state,
        jvp.lattice;
        samples=jvp.sampled_orbit,
        threaded=workspace.sensitivity.threaded,
    )
    for direction in 1:C
        parameter = first_parameter + direction - 1
        parameter > N_QUADS && break
        for factor in 1:problem.nfactors
            total = 0.0
            first_row = problem.nexperiments * (factor - 1) + 1
            last_row = first_row + problem.nexperiments - 1
            for bpm in 1:N_BPMS, row in first_row:last_row
                tangent_mm = 1_000.0 * ForwardDiff.partials(
                    jvp.sampled_orbit[row, bpm], direction
                )
                total += primal.residual_weight[row, bpm] * tangent_mm
            end
            gradient[factor, parameter] = total
        end
    end
    gradient
end

const _CPUProblem = Union{BoosterBatchProblem,BoosterFactorBatchProblem}

function _with_workspace(f, problem)
    problem.busy && error("$(typeof(problem)) is already in use; create one problem per chain")
    problem.busy = true
    try
        f()
    finally
        problem.busy = false
    end
end

function _prepare_factors!(problem::BoosterBatchProblem, factors::AbstractVector)
    length(factors) == N_QUADS || throw(DimensionMismatch(
        "quad_factors must contain $N_QUADS values"
    ))
    all(isfinite, factors) || throw(ArgumentError("quad_factors must be finite"))
    _update_quadrupoles!(problem.primal.quad_active, problem.prepared.quad_k1_base, factors)
end

function _prepare_factors!(problem::BoosterFactorBatchProblem, factors::AbstractMatrix)
    _validate_factor_matrix(problem, factors)
    workspace = problem.workspace
    _update_factor_batch_quadrupoles!(
        workspace.primal.quad_active, workspace.prepared.quad_k1_base,
        factors, problem.nexperiments,
    )
end

_solve_orbit!(problem::BoosterBatchProblem) = _solve_updated_primal!(problem)
_solve_orbit!(problem::BoosterFactorBatchProblem) =
    _solve_updated_primal!(problem.workspace)
_solve_orbit!(problem::_CPUProblem, factors) = _solve_orbit!(problem)

function _likelihood!(problem::BoosterBatchProblem)
    prepared, primal = problem.prepared, problem.primal
    predictions = _predict_primal!(problem)
    weighted_sum_of_squares = 0.0
    for index in eachindex(predictions)
        residual = prepared.observed_orbit_mm[index] - predictions[index]
        weight = residual * prepared.inv_variance_mm[index]
        primal.residual_weight[index] = weight
        weighted_sum_of_squares += residual * weight
    end
    prepared.loglik_constant - 0.5 * weighted_sum_of_squares
end

function _likelihood!(problem::BoosterFactorBatchProblem)
    workspace = problem.workspace
    prepared, primal = workspace.prepared, workspace.primal
    predictions = _predict_primal!(workspace)
    values = fill(Float64(problem.prepared.loglik_constant), problem.nfactors)
    for factor in 1:problem.nfactors
        weighted_sum_of_squares = 0.0
        first_row = problem.nexperiments * (factor - 1) + 1
        last_row = first_row + problem.nexperiments - 1
        for bpm in 1:N_BPMS, row in first_row:last_row
            residual = prepared.observed_orbit_mm[row, bpm] - predictions[row, bpm]
            weight = residual * prepared.inv_variance_mm[row, bpm]
            primal.residual_weight[row, bpm] = weight
            weighted_sum_of_squares += residual * weight
        end
        values[factor] -= 0.5 * weighted_sum_of_squares
    end
    values
end

function _gradient!(problem::BoosterBatchProblem, factors)
    gradient = zeros(Float64, N_QUADS)
    for first_parameter in 1:problem.sensitivity.chunksize:N_QUADS
        _accumulate_gradient_chunk!(gradient, problem, factors, first_parameter)
    end
    gradient
end

function _gradient!(problem::BoosterFactorBatchProblem, factors)
    gradient = zeros(Float64, problem.nfactors, N_QUADS)
    for first_parameter in 1:problem.workspace.sensitivity.chunksize:N_QUADS
        _accumulate_factor_gradient_chunk!(gradient, problem, factors, first_parameter)
    end
    gradient
end

function _evaluate_core!(problem, factors, ::Val{G}) where {G}
    _with_workspace(problem) do
        _prepare_factors!(problem, factors)
        _solve_orbit!(problem, factors)
        values = _likelihood!(problem)
        G ? (values, _gradient!(problem, factors)) : values
    end
end

"""
    predict_orbits(problem, quad_factors)

Return the batched selected-BPM orbit in millimetres. This posterior-predictive
path performs no quadrupole sensitivity calculation.
"""
function predict_orbits(
    problem::BoosterBatchProblem,
    quad_factors::AbstractVector{<:Real},
)
    _with_workspace(problem) do
        _prepare_factors!(problem, quad_factors)
        _solve_orbit!(problem)
        copy(_predict_primal!(problem))
    end
end

"""
    value_and_gradient!(problem, quad_factors)

Evaluate the scalar Gaussian orbit log likelihood and its gradient. The
gradient is reduced during each forward-JVP chunk; no `(33S) × 48` prediction
Jacobian is formed and the Newton iterations are never differentiated.
"""
function value_and_gradient!(
    problem::BoosterBatchProblem,
    quad_factors::AbstractVector{<:Real},
)
    _evaluate_core!(problem, quad_factors, Val(true))
end

"""
    factor_batch_value_and_gradient!(problem, quad_factors)

Evaluate independent likelihoods and gradients for every row of
`quad_factors`, whose required shape is `(nfactors, 48)`. The returned objects
have shapes `(nfactors,)` and `(nfactors, 48)`. Each row is exactly the result
of evaluating the original scalar problem against the fixed experiments.
"""
function factor_batch_value_and_gradient!(
    problem::BoosterFactorBatchProblem,
    quad_factors::AbstractMatrix{<:Real},
)
    _evaluate_core!(problem, quad_factors, Val(true))
end

value_and_gradient!(
    problem::BoosterFactorBatchProblem,
    quad_factors::AbstractMatrix{<:Real},
) = factor_batch_value_and_gradient!(problem, quad_factors)

booster_factor_batch_loglikelihoods(
    problem::BoosterFactorBatchProblem,
    quad_factors::AbstractMatrix{<:Real},
) = first(factor_batch_value_and_gradient!(problem, quad_factors))

function ChainRulesCore.rrule(
    ::typeof(booster_factor_batch_loglikelihoods),
    problem::BoosterFactorBatchProblem,
    quad_factors::AbstractMatrix{<:Real},
)
    loglikelihoods, gradient = factor_batch_value_and_gradient!(
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

"""
    booster_loglikelihood(problem, quad_factors)

Scalar inference primitive. Its ChainRules pullback uses the chunked forward
gradient computed by `value_and_gradient!`; reverse AD never enters tracking or
the closed-orbit Newton solver.
"""
function booster_loglikelihood(
    problem::BoosterBatchProblem,
    quad_factors::AbstractVector{<:Real},
)
    first(value_and_gradient!(problem, quad_factors))
end

function ChainRulesCore.rrule(
    ::typeof(booster_loglikelihood),
    problem::BoosterBatchProblem,
    quad_factors::AbstractVector{<:Real},
)
    loglikelihood, gradient = value_and_gradient!(problem, quad_factors)
    function pullback(loglikelihood_tangent)
        scale = unthunk(loglikelihood_tangent)
        factor_tangent = scale isa AbstractZero ?
            ZeroTangent() :
            ProjectTo(quad_factors)(scale .* gradient)
        NoTangent(), NoTangent(), factor_tangent
    end
    loglikelihood, pullback
end

ReverseDiff.@grad_from_chainrules booster_loglikelihood(
    problem::BoosterBatchProblem,
    quad_factors::ReverseDiff.TrackedArray,
)
ReverseDiff.@grad_from_chainrules booster_loglikelihood(
    problem::BoosterBatchProblem,
    quad_factors::AbstractVector{<:ReverseDiff.TrackedReal},
)

function lognormal_quad_prior(log_std; center::Symbol=:median)
    stds = log_std isa Real ? fill(Float64(log_std), N_QUADS) : Float64.(log_std)
    length(stds) == N_QUADS || throw(DimensionMismatch(
        "log_std must contain $N_QUADS values"
    ))
    all(value -> isfinite(value) && value > 0, stds) ||
        throw(ArgumentError("log_std values must be positive and finite"))
    locations = center === :median ? zeros(N_QUADS) :
        center === :mean ? -0.5 .* abs2.(stds) :
        throw(ArgumentError("center must be :median or :mean"))
    Turing.product_distribution(Turing.LogNormal.(locations, stds))
end

Turing.@model function booster_model(
    problem::BoosterBatchProblem,
    quad_prior=lognormal_quad_prior(0.03),
)
    quad_factors ~ quad_prior
    Turing.@addlogprob! booster_loglikelihood(problem, quad_factors)
end
