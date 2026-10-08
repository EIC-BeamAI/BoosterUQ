_realtype(::Type{T}) where {T<:Real} = T
_realtype(::Type{<:ForwardDiff.Dual{Tag,T}}) where {Tag,T} = T
_reference(lattice, state) = (;
    species=lattice.species_ref,
    p_over_q_ref=lattice.p_over_q_ref,
    t_ref=lattice.p_over_q_ref isa BatchParam ?
        BatchParam(zero(_realtype(eltype(state)))) : zero(_realtype(eltype(state))),
)

function _track!(state, lattice; samples=nothing, threaded=false, lost=nothing)
    reference = _reference(lattice, state)
    bunch = Bunch(state; reference...)
    for (element_index, element) in enumerate(lattice.line)
        BeamTracking.track!(
            bunch, element; rf_on=false, use_cpu_multithreading=threaded,
            use_explicit_SIMD=!(
                eltype(state) <: ForwardDiff.Dual ||
                KernelAbstractions.get_backend(state) isa KernelAbstractions.GPU
            ),
        )
        isnothing(samples) && continue
        row = BPM_ROW_BY_ELEMENT[element_index]
        iszero(row) && continue
        @views samples[:, row] .= bunch.coords.v[:, BPM_PLANE_BY_ELEMENT[element_index]]
    end
    if isnothing(lost)
        _check_alive!(bunch, KernelAbstractions.get_backend(state))
    else
        lost .|= bunch.state .!= BeamTracking.STATE_ALIVE
    end
    state
end

_check_alive!(bunch, ::KernelAbstractions.GPU) = nothing
_check_alive!(bunch, backend) =
    all(==(BeamTracking.STATE_ALIVE), bunch.state) ||
        error("a batched Booster particle was lost during tracking")

function _solve_updated_primal!(problem)
    workspace = problem.primal
    sensitivity = problem.sensitivity
    sensitivity.warm_start || fill!(workspace.closed_orbit_guess, 0.0)
    copyto!(workspace.closed_orbit, workspace.closed_orbit_guess)
    result = SciBmad.find_closed_orbit(
        workspace.lattice;
        v0=workspace.closed_orbit,
        coasting_beam=true,
        batch=Val(true),
        rf_on=false,
        warn=false,
        abstol=sensitivity.abstol,
        reltol=0.0,
        maxiter=sensitivity.maxiter,
    )
    all(==(SciBmad.RETCODE_SUCCESS), result.sol.retcode) ||
        error("closed orbit failed: $(result.sol.retcode)")
    sensitivity.warm_start && (workspace.closed_orbit_guess .= result.v0)

    # SciBmad's residual is x-M(x), so its AutoBatch Jacobian is already I-R.
    nzval = result.sol.jac.nzval
    nsettings = size(workspace.closed_orbit, 1)
    for setting in 1:nsettings, row in 1:4, column in 1:4
        workspace.fixed_point_lhs[setting, row, column] =
            nzval[row + 4 * (setting - 1) + 4 * nsettings * (column - 1)]
    end
    workspace
end


function _solve_primal!(problem, quad_factors)
    _update_quadrupoles!(
        problem.primal.quad_active,
        problem.prepared.quad_k1_base,
        quad_factors,
    )
    _solve_updated_primal!(problem)
end

function _solve_factor_primal!(problem::BoosterFactorBatchProblem, quad_factors)
    workspace = problem.workspace
    _update_factor_batch_quadrupoles!(
        workspace.primal.quad_active,
        workspace.prepared.quad_k1_base,
        quad_factors,
        problem.nexperiments,
    )
    _solve_updated_primal!(workspace)
end

function _predict_primal!(problem)
    workspace = problem.primal
    workspace.tracking_state .= workspace.closed_orbit
    _track!(
        workspace.tracking_state,
        workspace.lattice;
        samples=workspace.predicted_orbit_mm,
        threaded=problem.sensitivity.threaded,
    )
    workspace.predicted_orbit_mm .*= 1_000.0
    workspace.predicted_orbit_mm
end
