Base.@kwdef struct BatchedForwardSensitivity
    chunksize::Int = 8
    abstol::Float64 = 1e-12
    maxiter::Int = 30
    warm_start::Bool = true
    threaded::Bool = false
end

mutable struct _PrimalWorkspace{L,M,A}
    lattice::L
    quad_active::M
    closed_orbit_guess::M
    closed_orbit::M
    tracking_state::M
    predicted_orbit_mm::M
    residual_weight::M
    fixed_point_lhs::A
end

struct _BoosterJVPTag end
_jvp_tag(::Type{T}) where {T<:AbstractFloat} =
    typeof(ForwardDiff.Tag(_BoosterJVPTag(), T))

mutable struct _JVPWorkspace{C,D,L,M,V,A}
    lattice::L
    quad_active::M
    quad_seed::V
    tracking_state::M
    sampled_orbit::M
    closed_orbit_response::A
end

mutable struct BoosterBatchProblem{P,W,J,S}
    prepared::P
    primal::W
    jvp::J
    sensitivity::S
    busy::Bool
end

"""
    BoosterFactorBatchProblem

Workspace for evaluating multiple independent quadrupole-scaling vectors
against one fixed collection of Booster experiments. Internally the tracking
batch is the Cartesian product `(factor set, experiment)`, ordered with all
experiments for one factor set contiguous. ForwardDiff lanes represent the
same local quadrupole columns independently in every factor set; they never
couple different factor-set rows.
"""
mutable struct BoosterFactorBatchProblem{P,E,W}
    prepared::P
    expanded_prepared::E
    workspace::W
    nexperiments::Int
    nfactors::Int
    busy::Bool
end

@inline _batch_column(matrix, column) = BatchParam(@view matrix[:, column])

function _set_normal!(element, value, order; integrated)
    element.BMultipoleParams = _BoosterLatticeTemplate._normal_multipole(
        value, order; normalized=true, integrated
    )
end

function _set_skew!(element, value, order; integrated)
    element.BMultipoleParams = _BoosterLatticeTemplate._skew_multipole(
        value, order; normalized=true, integrated
    )
end

function _disable_apertures!(lattice)
    for element in lattice.line
        isnothing(element.ApertureParams) && continue
        element.x1_limit = -1.0e6
        element.x2_limit = 1.0e6
        element.y1_limit = -1.0e6
        element.y2_limit = 1.0e6
        element.aperture_shape = ApertureShape.Rectangular
    end
    lattice
end

function _build_batched_lattice(prepared, quad_active)
    template = _BoosterLatticeTemplate.booster
    reference = BatchParam(prepared.p_over_q_ref)
    if prepared.p_over_q_ref isa Array
        lattice = Beamline(
            deepcopy(collect(template.line));
            species_ref=template.species_ref,
            p_over_q_ref=reference,
        )
    else
        # Beamlines' public reference setter compares a scalar sentinel against
        # BatchParam data. On a GPU backend that comparison would launch a
        # reduction containing Float64 and cannot compile on GPU backends. Construct
        # with a representable scalar, then install the already-validated batch
        # reference in the untyped InitialBeamlineParams storage on the host.
        lattice = Beamline(
            deepcopy(collect(template.line));
            species_ref=template.species_ref,
            p_over_q_ref=one(eltype(prepared.p_over_q_ref)),
        )
        initial = getfield(first(lattice.line), :pdict)[InitialBeamlineParams]
        setfield!(initial, :ref_meaning, Beamlines.RefMeaning.p_over_q_ref)
        setfield!(initial, :ref, reference)
    end
    if eltype(prepared.p_over_q_ref) === Float32
        for element in lattice.line
            element.L = Float32(element.L)
        end
    end
    _disable_apertures!(lattice)
    elements = Dict(Symbol(element.name) => element for element in lattice.line)

    dipole_group = _BoosterLatticeTemplate._normal_multipoles(
        (
            BatchParam(prepared.dipole_k0),
            BatchParam(prepared.dipole_k1),
            BatchParam(prepared.dipole_k2),
        ),
        (1, 2, 3);
        normalized=true,
        integrated=false,
    )
    for name in DIPOLE_NAMES
        elements[name].BMultipoleParams = dipole_group
    end
    for (column, name) in enumerate((H_QUAD_NAMES..., V_QUAD_NAMES...))
        _set_normal!(elements[name], _batch_column(quad_active, column), 2;
                     integrated=false)
    end
    for (column, name) in enumerate((H_SEXTUPOLE_NAMES..., V_SEXTUPOLE_NAMES...))
        _set_normal!(elements[name], _batch_column(prepared.sext_k2, column), 3;
                     integrated=false)
    end
    for (column, name) in enumerate(H_CORRECTOR_NAMES)
        _set_normal!(
            elements[name], _batch_column(prepared.hcorrector_k0L, column), 1;
            integrated=true,
        )
    end
    for (column, name) in enumerate(V_CORRECTOR_NAMES)
        _set_skew!(elements[name], _batch_column(prepared.vcorrector_k0L, column), 1;
                   integrated=true)
    end
    _set_normal!(elements[:ACQA4], BatchParam(prepared.ac_quad_k1), 2;
                 integrated=false)
    for (column, name) in enumerate((:IJKDHC1, :IJKDHC3, :IJKDHC7, :IJKDHD1))
        _set_normal!(elements[name], _batch_column(prepared.injection_k0L, column), 1;
                     integrated=true)
    end
    for name in (:X1DHF3, :X2DHF3, :X3DHF3, :X4DHF3)
        _set_normal!(elements[name], BatchParam(prepared.extraction_f3_k0L), 1;
                     integrated=true)
    end
    _set_normal!(elements[:SPTMD3], BatchParam(prepared.extraction_d3_k0L), 1;
                 integrated=true)
    lattice
end

function _jvp_workspace(prepared, ::Val{C}) where {C}
    C > 0 || throw(ArgumentError("chunksize must be positive"))
    T = eltype(prepared.quad_k1_base)
    D = ForwardDiff.Dual{_jvp_tag(T),T,C}
    quad_active = _backend_zeros(prepared.quad_k1_base, D, length(prepared), N_QUADS)
    lattice = _build_batched_lattice(prepared, quad_active)
    tracking_state = _backend_zeros(prepared.quad_k1_base, D, length(prepared), 6)
    sampled_orbit = _backend_zeros(
        prepared.quad_k1_base, D, length(prepared), N_BPMS
    )
    closed_orbit_response = _backend_zeros(
        prepared.quad_k1_base, T, length(prepared), 4, C
    )
    _JVPWorkspace{C,D,typeof(lattice),typeof(quad_active),Vector{D},
                  typeof(closed_orbit_response)}(
        lattice,
        quad_active,
        fill(zero(D), N_QUADS),
        tracking_state,
        sampled_orbit,
        closed_orbit_response,
    )
end

function _backend_zeros(like, ::Type{T}, dimensions...) where {T}
    result = similar(like, T, dimensions...)
    fill!(result, zero(T))
    result
end

"""
    BoosterBatchProblem(prepared; sensitivity=BatchedForwardSensitivity())

Create the mutable primal and forward-sensitivity workspaces for one inference
chain. Construct a separate problem for each concurrently executing chain;
`prepared` may be shared.
"""
function BoosterBatchProblem(
    prepared::PreparedBoosterBatch;
    sensitivity::BatchedForwardSensitivity=BatchedForwardSensitivity(),
)
    sensitivity.chunksize > 0 || throw(ArgumentError("chunksize must be positive"))
    sensitivity.abstol > 0 && isfinite(sensitivity.abstol) ||
        throw(ArgumentError("abstol must be positive and finite"))
    sensitivity.maxiter > 0 || throw(ArgumentError("maxiter must be positive"))

    quad_active = copy(prepared.quad_k1_base)
    lattice = _build_batched_lattice(prepared, quad_active)
    nsettings = length(prepared)
    primal = _PrimalWorkspace(
        lattice,
        quad_active,
        _backend_zeros(quad_active, eltype(quad_active), nsettings, 6),
        _backend_zeros(quad_active, eltype(quad_active), nsettings, 6),
        _backend_zeros(quad_active, eltype(quad_active), nsettings, 6),
        _backend_zeros(quad_active, eltype(quad_active), nsettings, N_BPMS),
        _backend_zeros(quad_active, eltype(quad_active), nsettings, N_BPMS),
        _backend_zeros(quad_active, eltype(quad_active), nsettings, 4, 4),
    )
    jvp = _jvp_workspace(prepared, Val(sensitivity.chunksize))
    BoosterBatchProblem(prepared, primal, jvp, sensitivity, false)
end

_repeat_factor_rows(values::AbstractVector, nfactors) = repeat(values, nfactors)
_repeat_factor_rows(values::AbstractMatrix, nfactors) = repeat(values, nfactors, 1)

function _repeat_prepared_for_factors(
    prepared::PreparedBoosterBatch,
    nfactors::Int,
)
    nfactors > 0 || throw(ArgumentError("nfactors must be positive"))
    repeat_rows = values -> _repeat_factor_rows(values, nfactors)
    PreparedBoosterBatch(
        repeat_rows(prepared.p_over_q_ref),
        repeat_rows(prepared.dipole_k0),
        repeat_rows(prepared.dipole_k1),
        repeat_rows(prepared.dipole_k2),
        repeat_rows(prepared.quad_k1_base),
        repeat_rows(prepared.sext_k2),
        repeat_rows(prepared.hcorrector_k0L),
        repeat_rows(prepared.vcorrector_k0L),
        repeat_rows(prepared.ac_quad_k1),
        repeat_rows(prepared.injection_k0L),
        repeat_rows(prepared.extraction_f3_k0L),
        repeat_rows(prepared.extraction_d3_k0L),
        repeat_rows(prepared.observed_orbit_mm),
        repeat_rows(prepared.inv_variance_mm),
        nfactors * prepared.loglik_constant,
    )
end

"""
    BoosterFactorBatchProblem(prepared, nfactors; sensitivity=...)

Create a fixed-size factor-batch workspace. `prepared` contains only the
physical Booster experiments; `nfactors` is the independent number of
quadrupole-scaling vectors evaluated per call.
"""
function BoosterFactorBatchProblem(
    prepared::PreparedBoosterBatch,
    nfactors::Integer;
    sensitivity::BatchedForwardSensitivity=BatchedForwardSensitivity(),
)
    expanded = _repeat_prepared_for_factors(prepared, Int(nfactors))
    workspace = BoosterBatchProblem(expanded; sensitivity)
    BoosterFactorBatchProblem(
        prepared,
        expanded,
        workspace,
        length(prepared),
        Int(nfactors),
        false,
    )
end

batch_problem(prepared::PreparedBoosterBatch, ::KernelAbstractions.CPU; kwargs...) =
    BoosterBatchProblem(prepared; kwargs...)
batch_problem(
    prepared::PreparedBoosterBatch, nfactors::Integer,
    ::KernelAbstractions.CPU; kwargs...,
) = BoosterFactorBatchProblem(prepared, nfactors; kwargs...)

function _update_quadrupoles!(active, base, factors)
    length(factors) == N_QUADS || throw(DimensionMismatch(
        "quad_factors must contain $N_QUADS values"
    ))
    for quad in 1:N_QUADS
        @views active[:, quad] .= base[:, quad] .* factors[quad]
    end
    active
end

function _validate_factor_matrix(problem::BoosterFactorBatchProblem, factors)
    size(factors) == (problem.nfactors, N_QUADS) || throw(DimensionMismatch(
        "quad_factors must have size ($(problem.nfactors), $N_QUADS)"
    ))
    all(isfinite, factors) || throw(ArgumentError("quad_factors must be finite"))
    factors
end

function _update_factor_batch_quadrupoles!(
    active,
    base,
    factors,
    nexperiments::Int,
)
    nfactors, nquads = size(factors)
    size(active) == (nexperiments * nfactors, nquads) ||
        throw(DimensionMismatch("factor-batch quadrupole workspace has the wrong size"))
    active_3d = reshape(active, nexperiments, nfactors, nquads)
    base_3d = reshape(base, nexperiments, nfactors, nquads)
    active_3d .= base_3d .* reshape(factors, 1, nfactors, nquads)
    active
end


"""Allocate one independent CPU workspace per inference chain."""
function chain_problems(prepared; nchains::Int=Threads.nthreads(), kwargs...)
    nchains > 0 || throw(ArgumentError("nchains must be positive"))
    [BoosterBatchProblem(prepared; kwargs...) for _ in 1:nchains]
end
