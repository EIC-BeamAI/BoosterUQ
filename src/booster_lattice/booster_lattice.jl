using Beamlines

# The Booster is six superperiods of eight cells.

# Most cells differ only in their sector/position names.
# The handful of real exceptions are kept together below,
# instead of being spread through 48 nearly identical arrays.

const BOOSTER_SECTORS = 'A':'F'
const BOOSTER_POSITIONS = 1:8

const inch = 0.0254
const LEND = 2.42
const RHO = 13.8656
const LENQH = 0.493
const LENQV = 0.504
const LENACQ = 0.17
const LENS = 0.1


const BOOSTER_DRIFT_LENGTHS = (
    L007=0.069600,
    L011=0.111825,
    L012=0.117325,
    L014=0.138050,
    L0218=0.217555,
    L028=0.276675,
    L02835=0.2835,
    L02869=0.2869,
    L029=0.289875,
    L030=0.293725,
    L031=0.295375,
    L031s=(0.295375 - LENACQ) / 2,
    L032=0.305784,
    L057=0.570400,
    L2757=2.756936,
    LDH=LEND,
    LDHH=LEND / 2,
    LDH1=1.5 - (34.89inch - 0.28),
    LDH2=0.32,
    LDH3=0.32,
    LDH4=0.0,
)

@inline _drift(name::Symbol, length) = LineElement(
    kind="Drift",
    name=String(name),
    L=length,
)
@inline _drift(name::Symbol) = _drift(
    name,
    getproperty(BOOSTER_DRIFT_LENGTHS, name),
)

@inline _marker(name::Symbol; kwargs...) = LineElement(;
    kind="Marker",
    name=String(name),
    kwargs...,
)
@inline _bpm(name::Symbol) = LineElement(
    kind="BPM",
    name=String(name),
)
@inline _hkicker(name::Symbol; kwargs...) = LineElement(;
    kind="HKicker",
    name=String(name),
    kwargs...,
)
@inline _vkicker(name::Symbol; kwargs...) = LineElement(;
    kind="VKicker",
    name=String(name),
    kwargs...,
)
@inline _quadrupole(name::Symbol, length) = LineElement(
    kind="Quadrupole",
    name=String(name),
    L=length,
)
@inline _sextupole(name::Symbol) = LineElement(
    kind="Sextupole",
    name=String(name),
    L=LENS,
)
@inline _dipole(name::Symbol) = LineElement(
    kind="SBend",
    name=String(name),
    L=LEND,
    angle=LEND/RHO,
)


function _cell_entrance(sector::Char, position::Int)
    key = Symbol(sector, position)

    key === :C1 && return LineElement[
        _drift(:L028), _hkicker(:IJKDHC1), _drift(:L030),
    ]
    key === :C4 && return LineElement[]
    key === :C6 && return LineElement[
        _drift(:L02835), _marker(:IJFOIL), _drift(:L02869),
    ]
    key === :C7 && return LineElement[
        _drift(:L028), _hkicker(:IJKDHC7), _drift(:L030),
    ]
    key === :D1 && return LineElement[
        _drift(:L028), _hkicker(:IJKDHD1), _drift(:L030),
    ]
    LineElement[_drift(:L057)]
end

function _missing_dipole_section(sector::Char, position::Int)
    key = Symbol(sector, position)

    key === :A3 && return LineElement[
        _drift(:LDHH), _marker(:CAVITY), _drift(:LDHH),
    ]
    key === :B6 && return LineElement[
        _drift(:LDHH), _marker(:SPTMB6), _drift(:LDHH),
    ]
    key === :C3 && return LineElement[
        _drift(:L032), _hkicker(:IJKDHC3), _drift(:L2757),
        _marker(:SPTMC3), _drift(:L0218),
    ]
    key === :D3 && return LineElement[
        _drift(:LDH1), _marker(:IPMV), _drift(:LDH2), _marker(:IPMSK),
        _drift(:LDH3), _marker(:IPMH),
        _hkicker(:SPTMD3; L=34.89inch),
        _drift(:LDH4),
    ]
    key === :D6 && return LineElement[
        _marker(:SPTMD6),
        _drift(:LDH),
    ]
    key === :F3 && return LineElement[
        _drift(:LEK1, 0.130 + 0.2731), _hkicker(:X1DHF3),
        _drift(:LEK2, 0.6096), _hkicker(:X2DHF3),
        _drift(:LEK3, 0.5004), _hkicker(:X3DHF3),
        _drift(:LEK4, 0.5391), _hkicker(:X4DHF3),
        _drift(:LEK5, 0.4978 - 0.130),
    ]
    key === :F6 && return LineElement[
        _marker(:START_OF_BTA), _marker(:SPTMF6), _drift(:LDH),
    ]
    LineElement[_drift(:LDH)]
end

function _cell_exit(sector::Char, position::Int)
    key = Symbol(sector, position)

    key === :C3 && return _missing_dipole_section(sector, position)

    if key === :A4
        return LineElement[
            _drift(:L031s), _quadrupole(:ACQA4, LENACQ),
            _drift(:L031s), _dipole(:DHA4),
        ]
    end

    tail = LineElement[]
    key === :F2 && push!(tail, _marker(:START_OF_BEXT))
    push!(tail, _drift(isodd(position) ? :L029 : :L031))

    if position in (3, 6)
        append!(tail, _missing_dipole_section(sector, position))
    else
        push!(tail, _dipole(Symbol("DH", sector, position)))
    end
    tail
end

function _booster_cell(sector::Char, position::Int)
    vertical = isodd(position)
    plane = vertical ? "V" : "H"
    label = Symbol(sector, position)

    cell = _cell_entrance(sector, position)
    push!(cell,
        vertical ? _vkicker(Symbol("DVC", label)) :
                   _hkicker(Symbol("DHC", label)),
        _drift(:L007),
        _sextupole(Symbol("S", plane, label)),
        _drift(:L014),
        _bpm(Symbol("PUE", plane, label)),
        _drift(vertical ? :L011 : :L012),
        _quadrupole(
            Symbol("Q", plane, label),
            vertical ? LENQV : LENQH,
        ),
    )
    append!(cell, _cell_exit(sector, position))
    cell
end

"""
    build_booster_lattice() -> Vector{LineElement}

Build a fresh, unpowered Booster lattice. The batched UQ compiler installs
strengths with the scalar, automatic-differentiation, or GPU value type needed
for a particular problem.
"""
function build_booster_lattice()
    lattice = LineElement[]
    for sector in BOOSTER_SECTORS, position in BOOSTER_POSITIONS
        append!(lattice, _booster_cell(sector, position))
    end
    lattice
end
