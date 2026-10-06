# SciBmad's residual is x - M(x), so its sparse AutoBatch Jacobian is I-R.
# Values have contiguous 4-row blocks for each lane and coordinate column.
@kernel function _scibmad_lhs_kernel!(fixed_lhs, nzval, n)
    lane, row, column = @index(Global, NTuple)
    @inbounds fixed_lhs[lane, row, column] =
        nzval[row + 4 * (lane - 1) + 4 * n * (column - 1)]
end

function _solve_scibmad_orbit!(problem::_CUDAProblem)
    primal = problem.device.primal
    sensitivity = problem.device.sensitivity
    sensitivity.warm_start || fill!(primal.closed_orbit_guess, 0)
    copyto!(primal.closed_orbit, primal.closed_orbit_guess)
    T = eltype(primal.closed_orbit)
    result = SciBmad.find_closed_orbit(
        primal.lattice;
        v0=primal.closed_orbit, coasting_beam=true, batch=Val(true),
        rf_on=false, warn=false, reltol=0.0,
        abstol=max(T(sensitivity.abstol), T === Float32 ? T(1e-5) : 8eps(T)),
        maxiter=sensitivity.maxiter,
    )
    backend = KernelAbstractions.get_backend(primal.closed_orbit)
    _scibmad_lhs_kernel!(backend)(
        primal.fixed_point_lhs, result.sol.jac.nzVal,
        size(primal.closed_orbit, 1); ndrange=size(primal.fixed_point_lhs),
    )
    problem.orbit.converged .= vec(result.sol.retcode) .== SciBmad.RETCODE_SUCCESS
    problem.orbit.failed .= vec(result.sol.retcode) .== SciBmad.RETCODE_FAILURE
    sensitivity.warm_start && copyto!(primal.closed_orbit_guess, primal.closed_orbit)
    primal
end

_solve_orbit!(problem::_CUDAProblem, factors) =
    _solve_orbit_eltype!(problem, factors, eltype(problem.device.primal.closed_orbit))
_solve_orbit_eltype!(problem::_CUDAProblem, factors, ::Type{Float64}) =
    _solve_scibmad_orbit!(problem)
_solve_orbit_eltype!(problem::_CUDAProblem, factors::CuArray, ::Type{Float32}) =
    _solve_device_orbit!(problem, Val(true))
_solve_orbit_eltype!(problem::_CUDAProblem, factors::AbstractArray, ::Type{Float32}) =
    _solve_device_orbit!(problem, Val(false))
