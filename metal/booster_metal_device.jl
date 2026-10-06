# Metal-specific BeamTracking kernel argument adaptation.
function UQ.BeamTracking.Adapt.adapt_structure(
    to::Metal.Adaptor, parameter::UQ.BeamTracking._LoweredBatchParam{N},
) where {N}
    UQ.BeamTracking._LoweredBatchParam{N}(
        UQ.BeamTracking.Adapt.adapt(to, parameter.batch)
    )
end

# BeamTracking's generated integrator checks `eltype(coords.v) == Float32`;
# coordinate AD makes that type `Dual{...,Float32}` and leaves its Yoshida
# coefficients as Float64. This method keeps the identical step sequence.
function UQ.BeamTracking.order_four_integrator!(
    i, coords::UQ.BeamTracking.Coords{
        <:Any,<:AbstractMatrix{ForwardDiff.Dual{Tag,Float32,N}}
    },
    ker, params, photon_params, ds_step, n_steps, edge_params,
    ::Val{fringe_in}, ::Val{fringe_out}, ::Val{optimized}, L,
) where {Tag,N,fringe_in,fringe_out,optimized}
    w0, w1 = optimized ?
        (Float32(-0.6579630871775028) * ds_step,
         Float32(0.4144907717943757) * ds_step) :
        (Float32(-1.7024143839193153) * ds_step,
         Float32(1.3512071919596578) * ds_step)
    if !isnothing(edge_params) && fringe_in
        UQ.BeamTracking.fringe!(i, coords, edge_params..., 1)
    end
    s = zero(ds_step)
    if !isnothing(photon_params)
        UQ.BeamTracking.stochastic_radiation!(
            i, coords, s, photon_params..., ds_step / 2
        )
    end
    for step in 1:n_steps
        if optimized
            ker(i, coords, s, params..., w1)
            s += w1
        end
        ker(i, coords, s, params..., w1); s += w1
        ker(i, coords, s, params..., w0); s += w0
        ker(i, coords, s, params..., w1); s += w1
        if optimized
            ker(i, coords, s, params..., w1)
            s += w1
        end
        if !isnothing(photon_params)
            scale = step == n_steps ? ds_step / 2 : ds_step
            UQ.BeamTracking.stochastic_radiation!(
                i, coords, s, photon_params..., scale
            )
        end
        if step < n_steps
            dt = UQ.BeamTracking.compute_dt_ref(s, ker, params)
            UQ.BeamTracking.execute_callbacks(i, coords, s, dt)
        end
    end
    if !isnothing(edge_params) && fringe_out
        UQ.BeamTracking.fringe!(i, coords, edge_params..., -1)
    end
    nothing
end

# BeamTracking's host-side kernel preparation occasionally combines a literal
# Float64 with a Float32 batch before its normal numeric lowering step.
function Base.:/(n::Float64, b::UQ.BatchParam)
    values = b.batch
    values isa AbstractArray{Float32} ?
        UQ.BatchParam(Float32(n) ./ values) :
        invoke(/, Tuple{Number,UQ.BatchParam}, n, b)
end

# ForwardDiff promotes a Float32 dual and a Float64 literal to a Float64 dual.
# BeamTracking still has a few physical constants in device kernel expressions.
for op in (:+, :-, :*, :/, :<, :<=, :>, :>=)
    @eval begin
        Base.$op(x::ForwardDiff.Dual{Tag,Float32,N}, y::Float64) where {Tag,N} =
            Base.$op(x, Float32(y))
        Base.$op(x::Float64, y::ForwardDiff.Dual{Tag,Float32,N}) where {Tag,N} =
            Base.$op(Float32(x), y)
    end
end

# BeamTracking's generated multipole field promotes the coefficient container
# types with coordinate Duals. On Metal that can select a Float64 Dual even
# though every physical coefficient is Float32. Keep this orbit-AD specialization
# in Float32 throughout the Horner recurrence.
@generated function UQ.BeamTracking.normalized_field(
    ms::M, knl::K, ksl::S,
    x::ForwardDiff.Dual{Tag,Float32,W},
    y::ForwardDiff.Dual{Tag,Float32,W}, excluding,
) where {M,K,S,Tag,W}
    n = length(M)
    D = ForwardDiff.Dual{Tag,Float32,W}
    quote
        by = ms[$n] != excluding && ms[$n] > 0 ?
            $D(knl[$n]) : zero(x)
        bx = ms[$n] != excluding && ms[$n] > 0 ?
            $D(ksl[$n]) : zero(x)
        $([quote
            normal = $D(knl[$j])
            skew = $D(ksl[$j])
            for order in (ms[$(j+1)] - 1):-1:max(ms[$j], 1)
                tmp = (by * x - bx * y) / order
                bx = (by * y + bx * x) / order
                by = tmp
                if order == ms[$j] && ms[$j] != excluding
                    by += normal
                    bx += skew
                end
            end
        end for j in n-1:-1:1]...)
        for order in (ms[1] - 1):-1:1
            tmp = (by * x - bx * y) / order
            bx = (by * y + bx * x) / order
            by = tmp
        end
        bx, by
    end
end

function UQ.BeamTracking.Adapt.adapt_structure(
    to::Metal.Adaptor, reference::UQ.BeamTracking.RefState{R},
) where {R}
    adapt = value -> UQ.BeamTracking.Adapt.adapt(to, value)
    UQ.BeamTracking.RefState{R}(
        adapt(reference.t_enter), adapt(reference.beta_gamma_enter),
        adapt(reference.t_exit), adapt(reference.beta_gamma_exit),
        adapt(reference.L), adapt(reference.g), adapt(reference.ds_step),
    )
end
