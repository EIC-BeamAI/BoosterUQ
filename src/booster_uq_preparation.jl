module _BoosterLatticeTemplate
include(joinpath(@__DIR__, "booster_lattice", "booster_run.jl"))
end

const H_QUAD_LABELS = _BoosterLatticeTemplate.H_QUAD_LABELS
const V_QUAD_LABELS = _BoosterLatticeTemplate.V_QUAD_LABELS
const H_QUAD_NAMES = _BoosterLatticeTemplate.H_QUAD_NAMES
const V_QUAD_NAMES = _BoosterLatticeTemplate.V_QUAD_NAMES
const H_SEXTUPOLE_NAMES = _BoosterLatticeTemplate.H_SEXTUPOLE_NAMES
const V_SEXTUPOLE_NAMES = _BoosterLatticeTemplate.V_SEXTUPOLE_NAMES
const H_CORRECTOR_NAMES = _BoosterLatticeTemplate.H_CORRECTOR_NAMES
const V_CORRECTOR_NAMES = _BoosterLatticeTemplate.V_CORRECTOR_NAMES
const CORRECTOR_NAMES = (H_CORRECTOR_NAMES..., V_CORRECTOR_NAMES...)
const DIPOLE_NAMES = _BoosterLatticeTemplate.DIPOLE_NAMES

const H_BPM_LABELS = (
    :A6, :A8, :B4, :B6, :C2, :C6, :D2, :D8, :E2, :E4, :E6, :E8,
    :F2, :F4, :F8,
)
const V_BPM_LABELS = (
    :A1, :A3, :A5, :A7, :B1, :B5, :B7, :C1, :C3, :C5, :D3, :D5,
    :E5, :E7, :F1, :F3, :F5, :F7,
)
const H_BPM_NAMES = Tuple(Symbol(:PUEH, label) for label in H_BPM_LABELS)
const V_BPM_NAMES = Tuple(Symbol(:PUEV, label) for label in V_BPM_LABELS)
const BPM_NAMES = (H_BPM_NAMES..., V_BPM_NAMES...)
const BPM_PLANES = (
    ntuple(_ -> 1, length(H_BPM_NAMES))...,
    ntuple(_ -> 3, length(V_BPM_NAMES))...,
)

function _make_bpm_element_layout(line)
    row_for_element = zeros(Int, length(line))
    plane_for_element = zeros(Int, length(line))
    rows = Dict(name => row for (row, name) in enumerate(BPM_NAMES))
    for (element_index, element) in enumerate(line)
        row = get(rows, Symbol(element.name), 0)
        iszero(row) && continue
        iszero(row_for_element[element_index]) || error(
            "BPM $(element.name) occurs more than once in the Booster beamline"
        )
        row_for_element[element_index] = row
        plane_for_element[element_index] = BPM_PLANES[row]
    end
    sort(filter(value -> !iszero(value), row_for_element)) ==
        collect(1:length(BPM_NAMES)) || error(
        "the Booster beamline does not contain every requested BPM exactly once"
    )
    row_for_element, plane_for_element
end

const BPM_ROW_BY_ELEMENT, BPM_PLANE_BY_ELEMENT =
    _make_bpm_element_layout(_BoosterLatticeTemplate.booster.line)

const N_QH = length(H_QUAD_NAMES)
const N_QV = length(V_QUAD_NAMES)
const N_QUADS = N_QH + N_QV
const N_HC = length(H_CORRECTOR_NAMES)
const N_VC = length(V_CORRECTOR_NAMES)
const N_SEXT = length(H_SEXTUPOLE_NAMES) + length(V_SEXTUPOLE_NAMES)
const N_BPMS = length(BPM_NAMES)

"""
    PreparedBoosterBatch

Immutable description of a set of measured Booster operating points. All
transfer functions and magnetic-rigidity normalizations have already been
evaluated. Rows are machine settings; quadrupole columns follow the horizontal
then vertical ordering defined by `H_QUAD_NAMES` and `V_QUAD_NAMES`.

The arrays are treated as read-only and can be shared by multiple independently
mutable `BoosterBatchProblem`s (one problem per concurrent Turing chain).
"""
struct PreparedBoosterBatch{V<:AbstractVector,M<:AbstractMatrix,T<:AbstractFloat}
    p_over_q_ref::V
    dipole_k0::V
    dipole_k1::V
    dipole_k2::V
    quad_k1_base::M
    sext_k2::M
    hcorrector_k0L::M
    vcorrector_k0L::M
    ac_quad_k1::V
    injection_k0L::M
    extraction_f3_k0L::V
    extraction_d3_k0L::V
    observed_orbit_mm::M
    inv_variance_mm::M
    loglik_constant::T
end

Base.length(prepared::PreparedBoosterBatch) = length(prepared.p_over_q_ref)

function _template_elements_by_name()
    Dict(Symbol(element.name) => element for element in _BoosterLatticeTemplate.booster.line)
end

function _validate_observations(observed_orbit_mm, noise_std_mm, nsettings)
    observed = Matrix{Float64}(observed_orbit_mm)
    size(observed) == (nsettings, N_BPMS) || throw(DimensionMismatch(
        "observed_orbit_mm must have size ($nsettings, $N_BPMS)"
    ))
    all(isfinite, observed) ||
        throw(ArgumentError("observed_orbit_mm must contain only finite values"))

    noise = noise_std_mm isa Real ?
        fill(Float64(noise_std_mm), size(observed)) :
        Matrix{Float64}(noise_std_mm)
    size(noise) == size(observed) || throw(DimensionMismatch(
        "noise_std_mm must be scalar or have the same size as observed_orbit_mm"
    ))
    all(value -> isfinite(value) && value > 0, noise) ||
        throw(ArgumentError("noise_std_mm must be positive and finite"))
    observed, noise
end

function _complete_operating_point(op)
    op isa NamedTuple || throw(ArgumentError(
        "each operating point must be a NamedTuple"
    ))
    complete = merge(_BoosterLatticeTemplate.default_booster_operating_point(), op)
    merge(complete, (; corrector_currents=Tuple(complete.corrector_currents)))
end

"""
    prepare_booster_batch(operating_points, observed_orbit_mm;
                          noise_std_mm, calibration=default_calibration)

Compile measured machine settings into normalized magnetic strengths. This is
the only path that evaluates the nonlinear Booster transfer functions; it is
intended to run once before HMC starts.

Each item in `operating_points` is a named tuple. Missing optional fields are
filled from `default_booster_operating_point()`.
"""
function prepare_booster_batch(
    operating_points::AbstractVector,
    observed_orbit_mm::AbstractMatrix;
    noise_std_mm,
    calibration=_BoosterLatticeTemplate.default_booster_calibration(),
)
    nsettings = length(operating_points)
    nsettings > 0 || throw(ArgumentError("at least one operating point is required"))
    observed, noise = _validate_observations(
        observed_orbit_mm, noise_std_mm, nsettings
    )

    template_elements = _template_elements_by_name()
    quad_names = (H_QUAD_NAMES..., V_QUAD_NAMES...)
    sext_names = (H_SEXTUPOLE_NAMES..., V_SEXTUPOLE_NAMES...)
    quad_lengths = Float64[template_elements[name].L for name in quad_names]
    sext_lengths = Float64[template_elements[name].L for name in sext_names]
    ac_quad_length = Float64(template_elements[:ACQA4].L)

    p_over_q_ref = Vector{Float64}(undef, nsettings)
    dipole_k0 = similar(p_over_q_ref)
    dipole_k1 = similar(p_over_q_ref)
    dipole_k2 = similar(p_over_q_ref)
    quad_k1_base = Matrix{Float64}(undef, nsettings, N_QUADS)
    sext_k2 = Matrix{Float64}(undef, nsettings, N_SEXT)
    hcorrector_k0L = Matrix{Float64}(undef, nsettings, N_HC)
    vcorrector_k0L = Matrix{Float64}(undef, nsettings, N_VC)
    ac_quad_k1 = similar(p_over_q_ref)
    injection_k0L = Matrix{Float64}(undef, nsettings, 4)
    extraction_f3_k0L = similar(p_over_q_ref)
    extraction_d3_k0L = similar(p_over_q_ref)

    for setting in eachindex(operating_points)
        op = _complete_operating_point(operating_points[setting])
        length(op.corrector_currents) == N_HC + N_VC || throw(DimensionMismatch(
            "operating point $setting must contain $(N_HC + N_VC) corrector currents"
        ))
        fields = _BoosterLatticeTemplate.booster_fields(op, calibration)
        rigidity = Float64(fields.dipole.p_over_q_ref)
        isfinite(rigidity) && !iszero(rigidity) || throw(ArgumentError(
            "operating point $setting has a non-finite or zero reference rigidity"
        ))

        p_over_q_ref[setting] = rigidity
        dipole_k0[setting] = Float64(fields.dipole.Bn0 / rigidity)
        dipole_k1[setting] = Float64(fields.dipole.Bn1 / rigidity)
        dipole_k2[setting] = Float64(fields.dipole.Bn2 / rigidity)
        for quad in 1:N_QUADS
            quad_k1_base[setting, quad] =
                Float64(fields.quadrupoles[quad] / (rigidity * quad_lengths[quad]))
        end
        for sextupole in 1:N_SEXT
            sext_k2[setting, sextupole] =
                Float64(fields.sextupoles[sextupole] /
                        (rigidity * sext_lengths[sextupole]))
        end
        for corrector in 1:N_HC
            hcorrector_k0L[setting, corrector] =
                Float64(fields.correctors[corrector] / rigidity)
        end
        for corrector in 1:N_VC
            vcorrector_k0L[setting, corrector] =
                Float64(fields.correctors[N_HC + corrector] / rigidity)
        end
        ac_quad_k1[setting] =
            Float64(fields.fast_magnets.acqa4 / (rigidity * ac_quad_length))
        injection_k0L[setting, 1] = Float64(fields.fast_magnets.ijkdhc1 / rigidity)
        injection_k0L[setting, 2] = Float64(fields.fast_magnets.ijkdhc3 / rigidity)
        injection_k0L[setting, 3] = Float64(fields.fast_magnets.ijkdhc7 / rigidity)
        injection_k0L[setting, 4] = Float64(fields.fast_magnets.ijkdhd1 / rigidity)
        extraction_f3_k0L[setting] = Float64(fields.fast_magnets.f3kick)
        extraction_d3_k0L[setting] = Float64(fields.fast_magnets.d03kick)
    end

    normalized_arrays = (
        p_over_q_ref, dipole_k0, dipole_k1, dipole_k2, quad_k1_base,
        sext_k2, hcorrector_k0L, vcorrector_k0L, ac_quad_k1,
        injection_k0L, extraction_f3_k0L, extraction_d3_k0L,
    )
    all(array -> all(isfinite, array), normalized_arrays) || throw(ArgumentError(
        "compiled Booster strengths must all be finite"
    ))

    inv_variance = 1.0 ./ abs2.(noise)
    loglik_constant = -sum(log, noise) - length(noise) * log(2pi) / 2
    PreparedBoosterBatch(
        p_over_q_ref,
        dipole_k0,
        dipole_k1,
        dipole_k2,
        quad_k1_base,
        sext_k2,
        hcorrector_k0L,
        vcorrector_k0L,
        ac_quad_k1,
        injection_k0L,
        extraction_f3_k0L,
        extraction_d3_k0L,
        observed,
        inv_variance,
        loglik_constant,
    )
end
