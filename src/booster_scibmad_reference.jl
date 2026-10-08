# SciBmad's residuals construct `Bunch(v_cache)` with scalar reference fields.
# Specialize on its matrix cache so BatchParam beamline references retain their
# type before BeamTracking checks the Bunch. This file can be dropped when
# SciBmad constructs its residual Bunch with the beamline reference directly.
import SciBmad
import KernelAbstractions
const _SCI_UQ = isdefined(@__MODULE__, :BoosterBatchedUQCUDA) ?
    BoosterBatchedUQCUDA.UQ : BoosterBatchedUQ

function SciBmad._co_res!(
    residual, coordinates, lattice::_SCI_UQ.Beamlines.Beamline,
    set_kernel!, sub_kernel!, cache::AbstractMatrix, rf_on,
)
    n = size(coordinates, 1)
    @assert length(residual) == 6n
    bunch = _SCI_UQ.BeamTracking.Bunch(cache; _SCI_UQ._reference(lattice, cache)...)
    SciBmad.BTBL.check_bl_bunch!(bunch, lattice, false)
    set_kernel!(residual, cache, coordinates, n; ndrange=n)
    KernelAbstractions.synchronize(KernelAbstractions.get_backend(cache))
    _SCI_UQ.BeamTracking.track!(bunch, lattice; scalar_params=true, rf_on)
    sub_kernel!(residual, cache, n, Val(false); ndrange=n)
    KernelAbstractions.synchronize(KernelAbstractions.get_backend(cache))
    residual
end

function SciBmad._co_res_coast!(
    residual, coordinates, lattice::_SCI_UQ.Beamlines.Beamline,
    set_kernel!, sub_kernel!, cache::AbstractMatrix,
    constants, rf_on,
)
    n = size(coordinates, 1)
    @assert length(residual) == 4n
    @assert n == size(cache, 1) == size(constants, 1)
    bunch = _SCI_UQ.BeamTracking.Bunch(cache; _SCI_UQ._reference(lattice, cache)...)
    SciBmad.BTBL.check_bl_bunch!(bunch, lattice, false)
    set_kernel!(residual, cache, constants, coordinates, n; ndrange=n)
    KernelAbstractions.synchronize(KernelAbstractions.get_backend(cache))
    _SCI_UQ.BeamTracking.track!(bunch, lattice; scalar_params=true, rf_on)
    sub_kernel!(residual, cache, n, Val(true); ndrange=n)
    KernelAbstractions.synchronize(KernelAbstractions.get_backend(cache))
    residual
end
