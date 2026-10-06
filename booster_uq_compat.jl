# Compatibility needed by BeamTracking's batched, dual-valued parameters.
if !hasmethod(Base.isapprox, Tuple{BatchParam,BatchParam})
    function Base.isapprox(a::BatchParam, b::BatchParam; kwargs...)
        result = isapprox.(a.batch, b.batch; kwargs...)
        result isa Bool ? result : all(result)
    end
end
ForwardDiff.Dual{Tag,V,N}(b::BatchParam) where {Tag,V,N} =
    BatchParam(ForwardDiff.Dual{Tag,V,N}.(b.batch))
BeamTracking.num_lower(::Type{<:ForwardDiff.Dual{Tag,V}}, x::Float64) where
    {Tag,V<:Union{Float16,Float32}} = convert(V, x)
BeamTracking.num_lower(::Type{D}, x::Tuple) where
    {Tag,V<:Union{Float16,Float32},N,D<:ForwardDiff.Dual{Tag,V,N}} =
    map(y -> BeamTracking.num_lower(D, y), x)

# BeamTracking's reference-energy formulas multiply a Float32 BatchParam by
# Float64 physical constants before `num_lower` runs. Metal cannot allocate
# that intermediate Float64 array.
function BeamTracking.R_to_beta_gamma(species::BeamTracking.Species, r::BatchParam)
    b = r.batch
    T = b isa AbstractArray ? eltype(b) : typeof(b)
    scale = T(BeamTracking.R_to_beta_gamma(species, one(T)))
    BatchParam(b .* scale)
end
function BeamTracking.beta_gamma_to_v(bg::BatchParam)
    b = bg.batch
    T = b isa AbstractArray ? eltype(b) : typeof(b)
    BatchParam(T(BeamTracking.C_LIGHT) .* b ./ sqrt.(one(T) .+ b .^ 2))
end

# BeamTracking's generic multipole unpacker deliberately evaluates rotations,
# rigidity normalization, and length conversion branchlessly.  Every active
# multipole in the prepared Booster lattice is already normalized, has zero
# tilt, and is stored in the representation needed by its tracking kernel.
# Avoiding the dead arithmetic is especially important for BatchParams backed
# by wide ForwardDiff dual arrays: the generic `ifelse` path allocates and
# evaluates both sides even though the metadata flags are fixed.
const _BTBL_EXT = Base.get_extension(BeamTracking, :BeamTrackingBeamlinesExt)
isnothing(_BTBL_EXT) && error("BeamTracking's Beamlines extension is not loaded")
const NORMALIZED_BATCH_FASTPATH_ENABLED =
    get(ENV, "BOOSTER_UQ_DISABLE_NORMALIZED_FASTPATH", "0") != "1"
if NORMALIZED_BATCH_FASTPATH_ENABLED &&
        !isdefined(_BTBL_EXT, :_booster_normalized_batch_fastpath)
    Core.eval(_BTBL_EXT, quote
        const _booster_normalized_batch_fastpath = true

        @inline function get_strengths(
            bm::Beamlines.BMultipoleParams{BeamTracking.BatchParam,N},
            L,
            p_over_q_ref,
        ) where {N}
            if all(bm.normalized) && !any(bm.integrated) && all(iszero, bm.tilt)
                return make_static(bm.n), make_static(bm.s)
            end
            invoke(get_strengths, Tuple{Any,Any,Any}, bm, L, p_over_q_ref)
        end

        @inline function get_integrated_strengths(
            bm::Beamlines.BMultipoleParams{BeamTracking.BatchParam,N},
            L,
            p_over_q_ref,
        ) where {N}
            if all(bm.normalized) && all(bm.integrated) && all(iszero, bm.tilt)
                return make_static(bm.n), make_static(bm.s)
            end
            invoke(get_integrated_strengths, Tuple{Any,Any,Any}, bm, L, p_over_q_ref)
        end


        # `pure_bquadrupole` indexes the parameter container before strength
        # conversion, so its hot path receives the array-of-structures view.
        # The earlier container overloads remain useful for other element
        # routes, but do not intercept this call.
        @inline function get_strengths(
            bm::Beamlines.BMultipole{BeamTracking.BatchParam},
            L,
            p_over_q_ref,
        )
            tilt = getfield(bm.tilt, :batch)
            if bm.normalized && !bm.integrated &&
                    tilt isa Number && iszero(tilt)
                return bm.n, bm.s
            end
            invoke(get_strengths, Tuple{Any,Any,Any}, bm, L, p_over_q_ref)
        end

        @inline function get_integrated_strengths(
            bm::Beamlines.BMultipole{BeamTracking.BatchParam},
            L,
            p_over_q_ref,
        )
            tilt = getfield(bm.tilt, :batch)
            if bm.normalized && bm.integrated &&
                    tilt isa Number && iszero(tilt)
                return bm.n, bm.s
            end
            invoke(
                get_integrated_strengths,
                Tuple{Any,Any,Any},
                bm,
                L,
                p_over_q_ref,
            )
        end
    end)
end
