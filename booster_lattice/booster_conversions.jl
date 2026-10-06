using Beamlines


const V_QUAD_LABELS = Tuple(Symbol(sector, position)
                            for sector in 'A':'F' for position in (1, 3, 5, 7))
const H_QUAD_LABELS = Tuple(Symbol(sector, position)
                            for sector in 'A':'F' for position in (2, 4, 6, 8))
const V_QUAD_NAMES = Tuple(Symbol(:QV, label) for label in V_QUAD_LABELS)
const H_QUAD_NAMES = Tuple(Symbol(:QH, label) for label in H_QUAD_LABELS)
const V_SEXTUPOLE_NAMES = Tuple(Symbol(:SV, sector, position)
                                for sector in 'A':'F' for position in (1, 3, 5, 7))
const H_SEXTUPOLE_NAMES = Tuple(Symbol(:SH, sector, position)
                                for sector in 'A':'F' for position in (2, 4, 6, 8))
const V_CORRECTOR_NAMES = Tuple(Symbol(:DVC, label) for label in V_QUAD_LABELS)
const H_CORRECTOR_NAMES = Tuple(Symbol(:DHC, label) for label in H_QUAD_LABELS)
const DIPOLE_NAMES = Tuple(Symbol(:DH, sector, position)
                           for sector in 'A':'F' for position in (1, 2, 4, 5, 7, 8))
const QV_TRIM_LABELS = (:B5, :B7, :C1, :C3, :C5, :C7, :D1, :D3)
const QH_TRIM_LABELS = (:B6, :B8, :C2, :C4, :C6, :C8, :D2, :D4)

# Canonical field ordering used by the batched UQ compilation path.
const QUAD_SPECS = (
    ((:H, label) for label in H_QUAD_LABELS)...,
    ((:V, label) for label in V_QUAD_LABELS)...,
)
const SEXTUPOLE_NAMES = (H_SEXTUPOLE_NAMES..., V_SEXTUPOLE_NAMES...)

const _QH_TRIM_INDEX =
    Dict(label => index for (index, label) in enumerate(QH_TRIM_LABELS))
const _QV_TRIM_INDEX =
    Dict(label => index for (index, label) in enumerate(QV_TRIM_LABELS))

# Coefficients are stored in ascending polynomial order. Dipole fits use
# x = current / dipole_current_scale, keeping coefficient magnitudes uniform.
const DEFAULT_BOOSTER_CALIBRATION = (
    dipole_current_scale=2000.0,
    dipole_field_coefficients=(
        0.0009122, 0.4742, 0.06868, -0.19296, 0.29376,
        -0.25216, 0.121024, -0.0300928, 0.00297728,
    ),
    dipole_bn1_coefficients=(
        -0.0032574662, -1.697965688, -2.612802404, 11.9468411312,
        -25.3060901088, 25.0977614336, -1.24143122816,
        -30.379763261952, 48.1862912264704, -46.0586315776,
        31.63947114496, -16.07332071424, 5.9698214912,
        -1.57176782848, 0.277349957632, -0.0293658066944,
        0.00140775325696,
    ),
    dipole_bn1_bdot_coefficients=(
        3.571, 5.04, -24.58, 56.904, -72.496,
        53.312, -22.5472, 5.08416, -0.472832,
    ),
    dipole_bn1_scale=1.0e-3,
    reference_bdot_coefficient=0.0,
    dipole_bn2_offset=-0.4438,
    dipole_bn2_bdot_coefficient=0.31764,
    dipole_bn2_ramp_scale=0.0,
    bend_radius=RHO,
    main_to_trim_turn_ratio=0.20022,
    horizontal_bdot_current=3.4,
    vertical_bdot_current=4.8,
    stopband_turn_ratio=0.4,
    horizontal_quad_coefficients=(
        0.001818, 9.080e-4, 6.657e-9, -7.225e-12, 3.239e-15, -5.07e-19,
    ),
    vertical_quad_coefficients=(
        0.002099, 9.257e-4, 1.164e-8, -1.046e-11, 4.057e-15, -5.75e-19,
    ),
    horizontal_quad_bdot_coefficient=0.00004179,
    vertical_quad_bdot_coefficient=0.000041942,
    vertical_ear_quad_bdot_coefficient=0.000062913,
    vertical_quad_scale=1.0030,
    sextupole_integrated_field_per_amp=0.013144,
    corrector_integrated_field_per_amp=0.0000975,
    ac_quad_integrated_field_per_amp=6.7e-3 * LENQV,
    injection_kicker_integrated_field_per_amp=0.016 / 1200,
)

default_booster_calibration() = DEFAULT_BOOSTER_CALIBRATION

const DEFAULT_BOOSTER_OPERATING_POINT = (
    idipo=1255.8,
    iqhc=209.2,
    iqvc=-117.4,
    bdot=0.0,
    ish=0.0,
    isv=0.0,
    isebc8f8=0.0,
    isebb4e4=0.0,
    iacqa4=0.0,
    ikhc1=0.0,
    ikhc3=0.0,
    ikhc7=0.0,
    ikhd1=0.0,
    f3kick=0.0,
    d03kick=0.0,
    qv_trim_currents=ntuple(_ -> 0.0, length(QV_TRIM_LABELS)),
    qh_trim_currents=ntuple(_ -> 0.0, length(QH_TRIM_LABELS)),
    qvstr1=0.0,
    qvstr2=0.0,
    qhstr1=0.0,
    qhstr2=0.0,
    corrector_currents=ntuple(_ -> 0.0,
        length(H_CORRECTOR_NAMES) + length(V_CORRECTOR_NAMES)),
)

default_booster_operating_point() = DEFAULT_BOOSTER_OPERATING_POINT

@inline _horner(x, coefficients::Tuple{C}) where {C} = coefficients[1]
@inline _horner(x, coefficients::Tuple) =
    coefficients[1] + x * _horner(x, Base.tail(coefficients))

@inline function bdip(idipo, calibration=DEFAULT_BOOSTER_CALIBRATION)
    x = idipo / calibration.dipole_current_scale
    _horner(x, calibration.dipole_field_coefficients)
end

@inline function b10(idipo, bdot=0, calibration=DEFAULT_BOOSTER_CALIBRATION)
    x = idipo / calibration.dipole_current_scale
    alpha = calibration.reference_bdot_coefficient * bdot / calibration.bend_radius
    calibration.dipole_bn1_scale * (
        _horner(x, calibration.dipole_bn1_coefficients) +
        alpha * _horner(x, calibration.dipole_bn1_bdot_coefficients)
    )
end

@inline function dipole_strengths(idipo, bdot=0,
                                  calibration=DEFAULT_BOOSTER_CALIBRATION)
    field = bdip(idipo, calibration)
    rigidity = field * calibration.bend_radius -
        calibration.reference_bdot_coefficient * bdot
    (
        Bn0=field,
        Bn1=b10(idipo, bdot, calibration),
        Bn2=(calibration.dipole_bn2_offset +
             calibration.dipole_bn2_bdot_coefficient *
             calibration.dipole_bn2_ramp_scale) * rigidity /
            calibration.bend_radius,
        p_over_q_ref=rigidity,
    )
end

@inline b1lh(current, calibration=DEFAULT_BOOSTER_CALIBRATION) =
    _horner(current, calibration.horizontal_quad_coefficients)
@inline b1lv(current, calibration=DEFAULT_BOOSTER_CALIBRATION) =
    _horner(current, calibration.vertical_quad_coefficients)

@inline effective_qh_current(idipo, iqhc, bdot,
                             calibration=DEFAULT_BOOSTER_CALIBRATION) =
    idipo + calibration.main_to_trim_turn_ratio *
        (iqhc + bdot * calibration.horizontal_bdot_current)

@inline effective_qv_current(idipo, iqvc, bdot,
                             calibration=DEFAULT_BOOSTER_CALIBRATION) =
    idipo + calibration.main_to_trim_turn_ratio *
        (-iqvc + bdot * calibration.vertical_bdot_current)

@inline function cblh(current, idipo, bdot=0,
                      calibration=DEFAULT_BOOSTER_CALIBRATION)
    (1 - calibration.horizontal_quad_bdot_coefficient *
         bdot / bdip(idipo, calibration)) * b1lh(current, calibration)
end

@inline function cblv(current, idipo, bdot=0,
                      calibration=DEFAULT_BOOSTER_CALIBRATION; ear=false)
    transient = ear ? calibration.vertical_ear_quad_bdot_coefficient :
                      calibration.vertical_quad_bdot_coefficient
    -(1 - transient * bdot / bdip(idipo, calibration)) *
        calibration.vertical_quad_scale * b1lv(current, calibration)
end

@inline function _label_coordinates(label::Symbol)
    text = String(label)
    length(text) == 2 || throw(ArgumentError("invalid Booster cell label $label"))
    Int(text[1]) - Int('A'), Int(text[2]) - Int('0')
end

@inline function stopband_current(op, plane::Symbol, label::Symbol)
    sector, position = _label_coordinates(label)
    sector_sign = iseven(sector) ? 1 : -1
    plane === :H && position in (2, 8) && return sector_sign * op.qhstr1
    plane === :H && position == 4 && return sector_sign * op.qhstr2
    plane === :H && position == 6 && return -sector_sign * op.qhstr2
    plane === :V && position in (1, 7) && return sector_sign * op.qvstr1
    plane === :V && position == 3 && return sector_sign * op.qvstr2
    plane === :V && position == 5 && return -sector_sign * op.qvstr2
    throw(ArgumentError("label $label does not belong to plane $plane"))
end

@inline function quad_integrated_gradient(op, calibration, plane::Symbol, label::Symbol)
    if plane === :H
        index = get(_QH_TRIM_INDEX, label, 0)
        trim = iszero(index) ? zero(first(op.qh_trim_currents)) :
                              op.qh_trim_currents[index]
        current = effective_qh_current(op.idipo, op.iqhc, op.bdot, calibration) +
            calibration.main_to_trim_turn_ratio * trim +
            calibration.stopband_turn_ratio * stopband_current(op, plane, label)
        return cblh(current, op.idipo, op.bdot, calibration)
    elseif plane === :V
        index = get(_QV_TRIM_INDEX, label, 0)
        trim = iszero(index) ? zero(first(op.qv_trim_currents)) :
                              op.qv_trim_currents[index]
        current = effective_qv_current(op.idipo, op.iqvc, op.bdot, calibration) +
            calibration.main_to_trim_turn_ratio * trim +
            calibration.stopband_turn_ratio * stopband_current(op, plane, label)
        return cblv(
            current, op.idipo, op.bdot, calibration; ear=label in (:D5, :F5)
        )
    end
    throw(ArgumentError("plane must be :H or :V"))
end

@inline function sextupole_integrated_field(name::Symbol, ish, isv,
                                            isebc8f8=0, isebb4e4=0,
                                            calibration=DEFAULT_BOOSTER_CALIBRATION)
    horizontal_current = name in (:SHB4, :SHE4) ? ish - isebb4e4 :
                         name in (:SHC8, :SHF8) ? ish + isebc8f8 : ish
    text = String(name)
    startswith(text, "SH") &&
        return calibration.sextupole_integrated_field_per_amp * horizontal_current
    startswith(text, "SV") &&
        return -calibration.sextupole_integrated_field_per_amp * isv
    throw(ArgumentError("invalid sextupole name $name"))
end

@inline corrector_integrated_field(current,
                                   calibration=DEFAULT_BOOSTER_CALIBRATION) =
    calibration.corrector_integrated_field_per_amp * current

struct BoosterFields{D,Q,S,C,F}
    dipole::D
    quadrupoles::Q
    sextupoles::S
    correctors::C
    fast_magnets::F
end

function booster_fields(op, calibration=DEFAULT_BOOSTER_CALIBRATION)
    dipole = dipole_strengths(op.idipo, op.bdot, calibration)
    quadrupoles = map(QUAD_SPECS) do (plane, label)
        quad_integrated_gradient(op, calibration, plane, label)
    end
    sextupoles = map(SEXTUPOLE_NAMES) do name
        sextupole_integrated_field(
            name, op.ish, op.isv, op.isebc8f8, op.isebb4e4, calibration
        )
    end
    correctors = map(op.corrector_currents) do current
        corrector_integrated_field(current, calibration)
    end
    fast_magnets = (
        acqa4=calibration.ac_quad_integrated_field_per_amp * op.iacqa4,
        ijkdhc1=calibration.injection_kicker_integrated_field_per_amp * op.ikhc1,
        ijkdhc3=calibration.injection_kicker_integrated_field_per_amp * op.ikhc3,
        ijkdhc7=calibration.injection_kicker_integrated_field_per_amp * op.ikhc7,
        ijkdhd1=calibration.injection_kicker_integrated_field_per_amp * op.ikhd1,
        f3kick=op.f3kick,
        d03kick=op.d03kick,
    )
    BoosterFields(dipole, quadrupoles, sextupoles, correctors, fast_magnets)
end

# Construct complete multipole groups for the batched lattice.

function _normal_multipoles(values::Tuple, orders::Tuple;
                            normalized::Bool, integrated::Bool)
    length(values) == length(orders) || throw(DimensionMismatch(
        "multipole values and orders must have equal lengths"))
    zeroes = map(x -> zero(eltype(x)), values)
    count = length(values)
    BMultipoleParams(
        collect(values), collect(zeroes), collect(zeroes), collect(orders),
        fill(normalized, count), fill(integrated, count),
    )
end

@inline _normal_multipole(value, order; normalized=false, integrated=true) =
    _normal_multipoles((value,), (order,); normalized, integrated)

function _skew_multipole(value, order; normalized=false, integrated=true)
    zero_value = zero(eltype(value))
    BMultipoleParams(
        [zero_value], [value], [zero_value], [order],
        [normalized], [integrated],
    )
end
