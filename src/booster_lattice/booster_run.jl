using Beamlines
include("booster_lattice.jl")
include("booster_conversions.jl")
include("booster_setting.jl")

booster = Beamline(
    build_booster_lattice();
    species_ref=Species("proton"),
    p_over_q_ref=1.0,
)


#set_booster_apertures!(booster)
