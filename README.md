# BoosterUQ

BoosterUQ evaluates batched Booster closed orbits, BPM likelihoods, and
implicit gradients of 48 quadrupole calibration factors. It supports ordinary
experiment batches and a second batch axis for evaluating several independent
factor vectors against the same data.

## Documentation

- [API guide](docs/BOOSTER_UQ_API_GUIDE.md): input conventions, array shapes,
  CPU and CUDA usage, concurrency, and failure handling.
- [Backend workflow](docs/BOOSTER_UQ_BACKEND_WORKFLOW.md): source map,
  dispatch, device data flow, closed-orbit solving, and implicit gradients.
- [Benchmarks](benchmarks/): CUDA and Perlmutter CPU runners and reports.
- [Tests](tests/): centralized CUDA and Metal validation suites.

## Quick start

Prepare machine settings and observations once:

```julia
include("src/booster_batched_uq.jl")
using .BoosterBatchedUQ

prepared = prepare_booster_batch(
    operating_points,
    observed_orbit_mm;
    noise_std_mm,
)
```

Create one mutable problem per concurrently executing inference chain:

```julia
problem = BoosterBatchProblem(
    prepared;
    sensitivity=BatchedForwardSensitivity(chunksize=8),
)

factors = ones(N_QUADS)
predicted_orbit_mm = predict_orbits(problem, factors)
loglikelihood, gradient = value_and_gradient!(problem, factors)
model = booster_model(problem)
```

`PreparedBoosterBatch` is read-only and may be shared between problems. Warm
starts are enabled by default because each problem is intended to remain with
one inference chain.

## Independent factor vectors

Use `BoosterFactorBatchProblem` when several candidate factor vectors are
available together:

```julia
problem = BoosterFactorBatchProblem(
    prepared,
    size(factor_sets, 1);
    sensitivity=BatchedForwardSensitivity(chunksize=8),
)

loglikelihoods, gradients = factor_batch_value_and_gradient!(
    problem,
    factor_sets,
)
```

The backend tracks the Cartesian product of experiments and factor vectors.
Each factor row remains an independent 48-parameter problem.

## Backends

CPU and CUDA implement the same generic likelihood and gradient operations.
Backend dispatch is selected when constructing the problem:

```julia
using KernelAbstractions
cpu_problem = batch_problem(
    prepared,
    KernelAbstractions.CPU();
    chunksize=8,
)

include("src/cuda/booster_batched_uq_cuda.jl")
using .BoosterBatchedUQCUDA
using CUDA

gpu_problem = batch_problem(
    prepared,
    CUDA.CUDABackend();
    chunksize=4,
    device_eltype=Float64,
)
```

CUDA also accepts device factor arrays and can return the likelihood and
gradient without copying them to the host. See the API guide for the device
interface and recommendations for CPU processes, threads, and multiple GPUs.

The closed orbit is solved once per combined lane. Gradients use the implicit
fixed-point equation

```text
(I - R) dx/dq = dM/dq
```

and contract BPM tangents directly into the 48-component result. The workflow
does not differentiate through Newton or construct the full BPM response
Jacobian.
