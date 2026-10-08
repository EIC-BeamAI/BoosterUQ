function set_booster_apertures!(beamline)
    for element in beamline.line
        if element.kind == "SBend"
            element.x1_limit = -0.08
            element.x2_limit = 0.08
            element.y1_limit = -0.033
            element.y2_limit = 0.033
            element.aperture_shape = ApertureShape.Rectangular
        elseif element.name != "SPTMD3" &&
                (element.kind in ("Drift", "Quadrupole", "Sextupole") ||
                 element.name == "SPTMD6")
            element.x1_limit = -0.0742
            element.x2_limit = 0.0742
            element.y1_limit = -0.0742
            element.y2_limit = 0.0742
        elseif element.name == "SPTMD3"
            element.x1_limit = -0.096
            element.x2_limit =  0.05
            element.y1_limit = -0.096
            element.y2_limit =  0.096
            element.aperture_shape=ApertureShape.Rectangular
        end
    end
    beamline
end

@inline _divide_magnet_strength(value, denominator) =
    iszero(value) ? value : value / denominator

function _set_normal_multipoles!(element, values::Tuple, orders::Tuple;
                                  normalized::Bool, integrated::Bool)
    if all(iszero, values)
        element.BMultipoleParams = nothing
    else
        element.BMultipoleParams = _normal_multipoles(
            values, orders; normalized, integrated
        )
    end
    return element
end

@inline function _set_normal_multipole!(element, value, order;
                                         normalized=false, integrated=true)
    _set_normal_multipoles!(
        element, (value,), (order,); normalized, integrated
    )
end

function _set_skew_multipole!(element, value, order;
                              normalized=false, integrated=true)
    if iszero(value)
        element.BMultipoleParams = nothing
    else
        element.BMultipoleParams = _skew_multipole(
            value, order; normalized, integrated
        )
    end
    return element
end


function set_booster_fields!(
    lattice::Beamline,
    fields::BoosterFields,
)
    elements = Dict(Symbol(element.name) => element for element in lattice.line)

    rigidity = fields.dipole.p_over_q_ref

    # Reference rigidity
    lattice.p_over_q_ref = rigidity

    # ------------------------------------------------------------------
    # Main dipoles
    # ------------------------------------------------------------------
    dipole_values = map(
        value -> _divide_magnet_strength(value, rigidity),
        (fields.dipole.Bn0, fields.dipole.Bn1, fields.dipole.Bn2),
    )

    for name in DIPOLE_NAMES
        _set_normal_multipoles!(
            elements[name], dipole_values, (1, 2, 3);
            normalized=true,
            integrated=false,
        )
    end

    # ------------------------------------------------------------------
    # Main quadrupoles
    # fields.quadrupoles follows H_QUAD_NAMES..., V_QUAD_NAMES...
    # and contains integrated fields.
    # ------------------------------------------------------------------
    quad_names = (H_QUAD_NAMES..., V_QUAD_NAMES...)

    for (i, name) in enumerate(quad_names)
        element = elements[name]
        k1 = _divide_magnet_strength(
            fields.quadrupoles[i], rigidity * element.L
        )

        _set_normal_multipole!(
            element, k1, 2;
            normalized=true,
            integrated=false,
        )
    end

    # ------------------------------------------------------------------
    # Main sextupoles
    # ------------------------------------------------------------------
    sext_names = (H_SEXTUPOLE_NAMES..., V_SEXTUPOLE_NAMES...)

    for (i, name) in enumerate(sext_names)
        element = elements[name]
        k2 = _divide_magnet_strength(
            fields.sextupoles[i], rigidity * element.L
        )

        _set_normal_multipole!(
            element, k2, 3;
            normalized=true,
            integrated=false,
        )
    end

    # ------------------------------------------------------------------
    # Horizontal correctors
    # ------------------------------------------------------------------
    for (i, name) in enumerate(H_CORRECTOR_NAMES)
        k0L = _divide_magnet_strength(fields.correctors[i], rigidity)

        _set_normal_multipole!(
            elements[name], k0L, 1;
            normalized=true,
            integrated=true,
        )
    end

    # ------------------------------------------------------------------
    # Vertical correctors
    # ------------------------------------------------------------------
    offset = length(H_CORRECTOR_NAMES)

    for (i, name) in enumerate(V_CORRECTOR_NAMES)
        k0L = _divide_magnet_strength(
            fields.correctors[offset + i], rigidity
        )

        _set_skew_multipole!(
            elements[name], k0L, 1;
            normalized=true,
            integrated=true,
        )
    end

    # ------------------------------------------------------------------
    # AC quadrupole
    # ------------------------------------------------------------------
    acqa4 = elements[:ACQA4]

    acqa4_k1 = _divide_magnet_strength(
        fields.fast_magnets.acqa4, rigidity * acqa4.L
    )
    _set_normal_multipole!(
        acqa4, acqa4_k1, 2;
        normalized=true,
        integrated=false,
    )

    # ------------------------------------------------------------------
    # Injection kickers
    # ------------------------------------------------------------------
    injection_names = (:IJKDHC1, :IJKDHC3, :IJKDHC7, :IJKDHD1)
    injection_fields = (
        fields.fast_magnets.ijkdhc1,
        fields.fast_magnets.ijkdhc3,
        fields.fast_magnets.ijkdhc7,
        fields.fast_magnets.ijkdhd1,
    )

    for (name, field) in zip(injection_names, injection_fields)
        k0L = _divide_magnet_strength(field, rigidity)
        _set_normal_multipole!(
            elements[name], k0L, 1;
            normalized=true,
            integrated=true,
        )
    end

    # ------------------------------------------------------------------
    # Extraction kickers
    #
    # Unlike the injection kickers, f3kick/d03kick are already in the
    # normalized representation expected by the lattice.
    # ------------------------------------------------------------------
    for name in (:X1DHF3, :X2DHF3, :X3DHF3, :X4DHF3)
        _set_normal_multipole!(
            elements[name], fields.fast_magnets.f3kick, 1;
            normalized=true,
            integrated=true,
        )
    end

    _set_normal_multipole!(
        elements[:SPTMD3], fields.fast_magnets.d03kick, 1;
        normalized=true,
        integrated=true,
    )

    return lattice
end
