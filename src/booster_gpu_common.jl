# Included by the Metal and CUDA frontends after `const UQ` is defined.
struct OrbitDeviceWorkspace{S,L,R,F,C}
    state::S
    lhs::L
    rhs::R
    failed::F
    converged::C
end

struct _OrbitTag end

function _orbit_workspace(device)
    orbit = device.primal.closed_orbit
    T = eltype(orbit)
    D = ForwardDiff.Dual{typeof(ForwardDiff.Tag(_OrbitTag(), T)),T,4}
    n = size(orbit, 1)
    OrbitDeviceWorkspace(
        similar(orbit, D), similar(orbit, T, n, 4, 4),
        similar(orbit, T, n, 4, 1), similar(orbit, Bool, n),
        similar(orbit, Bool, n),
    )
end

@generated function _orbit_dual(::Type{D}, value, column) where {D<:ForwardDiff.Dual}
    T = ForwardDiff.valtype(D)
    terms = [:(column == $i ? one($T) : zero($T)) for i in 1:4]
    :($D(value, ForwardDiff.Partials(($(terms...),))))
end

@kernel function _seed_orbit!(state, orbit)
    lane, column = @index(Global, NTuple)
    @inbounds state[lane, column] = _orbit_dual(eltype(state), orbit[lane, column], column)
end

@kernel function _orbit_linearization!(lhs, rhs, state, orbit)
    lane = @index(Global, Linear)
    @inbounds for row in 1:4
        value = state[lane, row]
        rhs[lane, row, 1] = ForwardDiff.value(value) - orbit[lane, row]
        for column in 1:4
            lhs[lane, row, column] = (row == column ? one(eltype(lhs)) : zero(eltype(lhs))) -
                                     ForwardDiff.partials(value, column)
        end
    end
end

@kernel function _accept_orbit!(converged, fixed_lhs, lhs, rhs, tolerance)
    lane = @index(Global, Linear)
    if !converged[lane]
        largest = zero(eltype(rhs))
        @inbounds for row in 1:4
            largest = max(largest, abs(rhs[lane, row, 1]))
        end
        if largest <= tolerance
            converged[lane] = true
            @inbounds for row in 1:4, column in 1:4
                fixed_lhs[lane, row, column] = lhs[lane, row, column]
            end
        end
    end
end

@kernel function _orbit_update!(orbit, step, converged, failed)
    lane, column = @index(Global, NTuple)
    if column <= 4 && !converged[lane] && !failed[lane]
        @inbounds orbit[lane, column] += step[lane, column, 1]
    end
end

@kernel function _row_solve!(lhs, rhs, failed, nstate, nparameters, skip)
    lane = @index(Global, Linear)
    if !(failed[lane] || (skip !== nothing && skip[lane]))
      @inbounds begin
        for column in 1:nstate
            pivot_row = column
            pivot_abs = abs(lhs[lane, column, column])
            for candidate in (column + 1):nstate
                a = abs(lhs[lane, candidate, column])
                if a > pivot_abs
                    pivot_abs, pivot_row = a, candidate
                end
            end
            if !(isfinite(pivot_abs) && pivot_abs > zero(pivot_abs))
                failed[lane] = true
                break
            end
            if pivot_row != column
                for j in column:nstate
                    lhs[lane, column, j], lhs[lane, pivot_row, j] =
                        lhs[lane, pivot_row, j], lhs[lane, column, j]
                end
                for j in 1:nparameters
                    rhs[lane, column, j], rhs[lane, pivot_row, j] =
                        rhs[lane, pivot_row, j], rhs[lane, column, j]
                end
            end
            pivot = lhs[lane, column, column]
            for i in (column + 1):nstate
                factor = lhs[lane, i, column] / pivot
                for j in (column + 1):nstate
                    lhs[lane, i, j] -= factor * lhs[lane, column, j]
                end
                for j in 1:nparameters
                    rhs[lane, i, j] -= factor * rhs[lane, column, j]
                end
            end
        end
        if !failed[lane]
            for parameter in 1:nparameters, i in nstate:-1:1
                total = rhs[lane, i, parameter]
                for j in (i + 1):nstate
                    total -= lhs[lane, i, j] * rhs[lane, j, parameter]
                end
                rhs[lane, i, parameter] = total / lhs[lane, i, i]
            end
        end
      end
    end
end

_solve_device_orbit!(problem) = _solve_device_orbit!(problem, Val(false))
_device_lost(problem, ::Val{false}) = nothing
_device_lost(problem, ::Val{true}) = problem.tracking_lost

function _solve_device_orbit!(problem, ::Val{DEVICE_ONLY}) where {DEVICE_ONLY}
    device, work = problem.device, problem.orbit
    primal = device.primal
    backend = KernelAbstractions.get_backend(primal.closed_orbit)
    device.sensitivity.warm_start || fill!(primal.closed_orbit_guess, 0)
    copyto!(primal.closed_orbit, primal.closed_orbit_guess)
    n = size(primal.closed_orbit, 1)
    T = eltype(primal.closed_orbit)
    tolerance = max(T(device.sensitivity.abstol), 8eps(T))
    fill!(work.converged, false)
    fill!(work.failed, false)
    for iteration in 1:device.sensitivity.maxiter
        _seed_orbit!(backend)(work.state, primal.closed_orbit; ndrange=size(work.state))
        UQ._track!(
            work.state, primal.lattice; threaded=false,
            lost=_device_lost(problem, Val(DEVICE_ONLY)),
        )
        _orbit_linearization!(backend)(
            work.lhs, work.rhs, work.state, primal.closed_orbit; ndrange=n,
        )
        _accept_orbit!(backend)(
            work.converged, primal.fixed_point_lhs, work.lhs, work.rhs,
            tolerance; ndrange=n,
        )
        if !DEVICE_ONLY && all(work.converged)
            device.sensitivity.warm_start &&
                copyto!(primal.closed_orbit_guess, primal.closed_orbit)
            return primal
        end
        if iteration < device.sensitivity.maxiter
            _row_solve!(backend)(
                work.lhs, work.rhs, work.failed, 4, 1, work.converged; ndrange=n,
            )
            !DEVICE_ONLY && any(work.failed) &&
                error("singular on-device closed-orbit Jacobian")
            _orbit_update!(backend)(
                primal.closed_orbit, work.rhs, work.converged, work.failed;
                ndrange=size(primal.closed_orbit),
            )
        end
    end
    DEVICE_ONLY || error("on-device closed orbit failed to converge in $(device.sensitivity.maxiter) iterations")
    device.sensitivity.warm_start &&
        copyto!(primal.closed_orbit_guess, primal.closed_orbit)
    primal
end

function _seed_fixed_device!(jvp, closed_orbit)
    D = eltype(jvp.tracking_state)
    jvp.tracking_state .= _zero_dual.(Ref(D), closed_orbit)
    jvp.tracking_state
end

_extract_and_solve_device!(problem) = _extract_and_solve_device!(problem, Val(false))

function _extract_and_solve_device!(problem, ::Val{DEVICE_ONLY}) where {DEVICE_ONLY}
    device = problem.device
    jvp = device.jvp
    response = jvp.closed_orbit_response
    state = jvp.tracking_state
    C = size(response, 3)
    for coordinate in 1:4, direction in 1:C
        @views response[:, coordinate, direction] .=
            _partial.(state[:, coordinate], Ref(Val(direction)))
    end
    copyto!(problem.lhs_work, device.primal.fixed_point_lhs)
    DEVICE_ONLY || fill!(problem.failed, false)
    backend = KernelAbstractions.get_backend(problem.lhs_work)
    _row_solve!(backend)(
        problem.lhs_work,
        response,
        problem.failed,
        4,
        C,
        nothing;
        ndrange=size(response, 1),
    )
    if !DEVICE_ONLY
        KernelAbstractions.synchronize(backend)
        failed = Array(problem.failed)
        any(failed) && error(
            "singular or non-finite on-device closed-orbit sensitivity for settings " *
            join(findall(failed), ", ")
        )
    end
    response
end

@generated function _factor_dual(
    ::Type{D}, value, derivative, parameter, first_parameter
) where {D<:ForwardDiff.Dual}
    T = ForwardDiff.valtype(D)
    C = ForwardDiff.npartials(D)
    partials = Expr(
        :tuple,
        [:(parameter == first_parameter + $(direction - 1) ?
            derivative : zero($T)) for direction in 1:C]...,
    )
    :($D(value, ForwardDiff.Partials($partials)))
end

@kernel function _seed_factor_quadrupoles_kernel!(
    active,
    base,
    factors,
    nexperiments,
    first_parameter,
)
    row, parameter = @index(Global, NTuple)
    factor = (row - 1) ÷ nexperiments + 1
    base_value = base[row, parameter]
    factor_value = factors[factor, parameter]
    active[row, parameter] = _factor_dual(
        eltype(active),
        base_value * factor_value,
        base_value,
        parameter,
        first_parameter,
    )
end

@generated function _dual_from_response(
    ::Type{D}, value, response, setting, coordinate
) where {D<:ForwardDiff.Dual}
    C = ForwardDiff.npartials(D)
    partials = Expr(
        :tuple,
        [:(response[setting, coordinate, $direction]) for direction in 1:C]...,
    )
    :($D(value, ForwardDiff.Partials($partials)))
end

@kernel function _seed_moving_kernel!(state, closed_orbit, response)
    setting, coordinate = @index(Global, NTuple)
    D = eltype(state)
    if coordinate <= 4
        state[setting, coordinate] = _dual_from_response(
            D,
            closed_orbit[setting, coordinate],
            response,
            setting,
            coordinate,
        )
    else
        state[setting, coordinate] = _zero_dual(
            D, closed_orbit[setting, coordinate]
        )
    end
end

_seed_moving_device!(jvp, closed_orbit) =
    _seed_moving_device!(jvp, closed_orbit, Val(false))

function _seed_moving_device!(jvp, closed_orbit, ::Val{DEVICE_ONLY}) where {DEVICE_ONLY}
    response = jvp.closed_orbit_response
    backend = KernelAbstractions.get_backend(jvp.tracking_state)
    _seed_moving_kernel!(backend)(
        jvp.tracking_state,
        closed_orbit,
        response;
        ndrange=size(jvp.tracking_state),
    )
    DEVICE_ONLY || KernelAbstractions.synchronize(backend)
    jvp.tracking_state
end

function _device_gradient_chunk!(gradient, problem, factors, first_parameter)
    device = problem.device
    prepared = device.prepared
    primal = device.primal
    jvp = device.jvp
    C = size(jvp.closed_orbit_response, 3)
    T = eltype(primal.tracking_state)

    UQ._seed_quadrupoles!(jvp, factors, first_parameter)
    UQ._update_quadrupoles!(jvp.quad_active, prepared.quad_k1_base, jvp.quad_seed)

    _seed_fixed_device!(jvp, primal.closed_orbit)
    UQ._track!(jvp.tracking_state, jvp.lattice; threaded=false)
    _extract_and_solve_device!(problem)

    _seed_moving_device!(jvp, primal.closed_orbit)
    UQ._track!(
        jvp.tracking_state,
        jvp.lattice;
        samples=jvp.sampled_orbit,
        threaded=false,
    )
    for direction in 1:C
        parameter = first_parameter + direction - 1
        parameter > UQ.N_QUADS && break
        tangent = _partial.(jvp.sampled_orbit, Ref(Val(direction)))
        gradient[parameter] = Float64(
            T(1_000) * sum(primal.residual_weight .* tangent)
        )
    end
    gradient
end

@kernel function _factor_gradient_kernel!(
    gradient,
    residual_weight,
    sampled_orbit,
    nexperiments,
    first_parameter,
)
    factor, direction = @index(Global, NTuple)
    parameter = first_parameter + direction - 1
    if parameter <= size(gradient, 2)
        T = eltype(gradient)
        total = zero(T)
        first_row = nexperiments * (factor - 1) + 1
        last_row = first_row + nexperiments - 1
        for bpm in axes(sampled_orbit, 2), row in first_row:last_row
            tangent = ForwardDiff.partials(sampled_orbit[row, bpm], direction)
            total += residual_weight[row, bpm] * tangent
        end
        gradient[factor, parameter] = T(1_000) * total
    end
end

_device_factor_gradient_chunk!(problem, first_parameter) =
    _device_factor_gradient_chunk!(problem, first_parameter, Val(false))

function _device_factor_gradient_chunk!(problem, first_parameter, ::Val{DEVICE_ONLY}) where {DEVICE_ONLY}
    device = problem.device
    prepared = device.prepared
    primal = device.primal
    jvp = device.jvp
    C = size(jvp.closed_orbit_response, 3)
    backend = KernelAbstractions.get_backend(jvp.quad_active)

    _seed_factor_quadrupoles_kernel!(backend)(
        jvp.quad_active,
        prepared.quad_k1_base,
        problem.factors_device,
        problem.nexperiments,
        first_parameter;
        ndrange=size(jvp.quad_active),
    )
    DEVICE_ONLY || KernelAbstractions.synchronize(backend)

    _seed_fixed_device!(jvp, primal.closed_orbit)
    UQ._track!(
        jvp.tracking_state, jvp.lattice; threaded=false,
        lost=_device_lost(problem, Val(DEVICE_ONLY)),
    )
    _extract_and_solve_device!(problem, Val(DEVICE_ONLY))

    _seed_moving_device!(jvp, primal.closed_orbit, Val(DEVICE_ONLY))
    UQ._track!(
        jvp.tracking_state,
        jvp.lattice;
        samples=jvp.sampled_orbit,
        threaded=false,
        lost=_device_lost(problem, Val(DEVICE_ONLY)),
    )
    _factor_gradient_kernel!(backend)(
        problem.gradient_device,
        primal.residual_weight,
        jvp.sampled_orbit,
        problem.nexperiments,
        first_parameter;
        ndrange=(problem.nfactors, C),
    )
    DEVICE_ONLY || KernelAbstractions.synchronize(backend)
    nothing
end
