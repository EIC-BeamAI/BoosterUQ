# Booster UQ API guide

This guide introduces the supported CPU and CUDA interfaces for evaluating the
Booster orbit likelihood and its derivatives. For the implementation data flow
and source-file map, see [BOOSTER_UQ_BACKEND_WORKFLOW.md](BOOSTER_UQ_BACKEND_WORKFLOW.md).

## 1. Concepts and array shapes

The interface has four independent forms of parallelism:

| Name | Meaning | Representation |
|---|---|---|
| Experiment | One measured machine setting and its 33 BPM readings | Row of `PreparedBoosterBatch` |
| Factor set | One independent vector of 48 quadrupole scale factors | Row of an `nfactors × 48` matrix |
| Derivative direction | A quadrupole derivative propagated in the current ForwardDiff pass | Dual width `chunksize` |
| Chain | A stateful inference sequence with its own warm-started orbit | One problem workspace |

Experiments share a factor vector in `BoosterBatchProblem`. A
`BoosterFactorBatchProblem` evaluates several independent factor vectors against
the same experiments. The factor and experiment axes form a Cartesian product;
they are not derivative dimensions.

The 48 inferred values are multiplicative corrections to the prepared main
quadrupole strengths:

```text
active_k1[experiment, quadrupole] =
    base_k1[experiment, quadrupole] * factor[quadrupole]
```

Use this selection as a starting point:

| Workload | Recommended mode |
|---|---|
| One factor vector, latency-sensitive | One CPU problem |
| Independent MCMC chains | Persistent single-thread CPU processes on physical cores |
| Several factor vectors available at the same time | CPU factor problem for modest batches; benchmark CUDA for large batches |
| Large factors × experiments Cartesian product already on GPU | CUDA factor problem with device inputs |
| Several NVIDIA GPUs | One Julia process per GPU, with factor sets or chains partitioned between processes |

The choice depends on both dimensions. Equal products of experiments and
factor sets can have different costs, and CUDA's high fixed launch cost makes
it unsuitable as a default single-chain accelerator.

## 2. Prepare a dataset once

The common preparation step runs on the CPU. It evaluates the Booster transfer
functions, normalizes fields by rigidity and element length, validates the BPM
data, and caches the Gaussian likelihood constants.

```julia
include("booster_batched_uq.jl")
using .BoosterBatchedUQ

operating_points = [
    (idipo=1255.8, iqhc=209.2, iqvc=-117.4,
     corrector_currents=ntuple(_ -> 0.0, length(CORRECTOR_NAMES))),
    (idipo=1260.0, iqhc=210.0, iqvc=-118.0,
     corrector_currents=ntuple(_ -> 0.0, length(CORRECTOR_NAMES))),
]
observed_orbit_mm = zeros(length(operating_points), 33)

prepared = prepare_booster_batch(
    operating_points,
    observed_orbit_mm;
    noise_std_mm=0.2,
)
```

Each operating point is a named tuple. Omitted fields are filled from
`default_booster_operating_point()`. Every operating point must ultimately
contain 48 corrector currents. `observed_orbit_mm` must have shape
`(nexperiments, 33)`. `noise_std_mm` may be one positive scalar or a matrix with
that same shape.

`PreparedBoosterBatch` is read-only after construction. CPU workspaces may
share it within a process. CUDA constructors copy it to persistent device
storage.

### 2.1 Corrector ordering and the notebook input names

The API accepts one flat `corrector_currents` tuple with 48 entries. Its order
is **all horizontal correctors first, then all vertical correctors**. These
exported constants are the source of truth:

```julia
CORRECTOR_NAMES == (H_CORRECTOR_NAMES..., V_CORRECTOR_NAMES...)
```

The exact positions are:

| Index | Horizontal name | Index | Vertical name |
|---:|---|---:|---|
| 1 | `DHCA2` | 25 | `DVCA1` |
| 2 | `DHCA4` | 26 | `DVCA3` |
| 3 | `DHCA6` | 27 | `DVCA5` |
| 4 | `DHCA8` | 28 | `DVCA7` |
| 5 | `DHCB2` | 29 | `DVCB1` |
| 6 | `DHCB4` | 30 | `DVCB3` |
| 7 | `DHCB6` | 31 | `DVCB5` |
| 8 | `DHCB8` | 32 | `DVCB7` |
| 9 | `DHCC2` | 33 | `DVCC1` |
| 10 | `DHCC4` | 34 | `DVCC3` |
| 11 | `DHCC6` | 35 | `DVCC5` |
| 12 | `DHCC8` | 36 | `DVCC7` |
| 13 | `DHCD2` | 37 | `DVCD1` |
| 14 | `DHCD4` | 38 | `DVCD3` |
| 15 | `DHCD6` | 39 | `DVCD5` |
| 16 | `DHCD8` | 40 | `DVCD7` |
| 17 | `DHCE2` | 41 | `DVCE1` |
| 18 | `DHCE4` | 42 | `DVCE3` |
| 19 | `DHCE6` | 43 | `DVCE5` |
| 20 | `DHCE8` | 44 | `DVCE7` |
| 21 | `DHCF2` | 45 | `DVCF1` |
| 22 | `DHCF4` | 46 | `DVCF3` |
| 23 | `DHCF6` | 47 | `DVCF5` |
| 24 | `DHCF8` | 48 | `DVCF7` |

This is made to agree with [sample.ipynb](sample.ipynb), cell 7:

- `cx_l` is `String.(H_CORRECTOR_NAMES)`;
- `cy_l` is `String.(V_CORRECTOR_NAMES)`; and
- `get_state_at` creates those names by converting `*-th-ps` devices to
  `DHC...` and `*-tv-ps` devices to `DVC...`.

All 48 positions are active. In particular, notebook values `DHCD6` and
`DHCF6` at indices 15 and 23 are applied to their corresponding lattice
elements.

Construct the API tuple by name rather than relying on dictionary iteration:

```julia
function operating_point_from_notebook_state(state::AbstractDict)
    value(name) = state[String(name)]
    (;
        idipo=value(:idipo),
        iqhc=value(:iqhc),
        iqvc=value(:iqvc),
        ish=value(:ish),
        isv=value(:isv),
        corrector_currents=Tuple(value(name) for name in CORRECTOR_NAMES),
    )
end

operating_points = operating_point_from_notebook_state.(notebook_states)
```

The notebook's `hcontrol` and `vcontrol` matrices cannot be passed directly.
Each contains 24 correctors from only one plane followed by the five static
currents `idipo`, `iqhc`, `iqvc`, `ish`, and `isv`. The API named tuple stores
those five currents as fields and combines both corrector planes into the
48-entry tuple above. After `getControlsFromData` has produced these matrices,
the opposite-plane corrector readbacks have already been discarded. Preserve
the original dictionaries returned by `get_state_at`, or change that extraction
step to retain both planes, before constructing API operating points.

### 2.2 Notebook integration differences

Several details require an explicit choice when connecting
[sample.ipynb](sample.ipynb):

1. **The real vertical scan perturbs only a subset.** Notebook cell 74 defines
   `vcors` as `A1, C7, D1, D3, D5, D7, E1, E3, E5, E7, F1, F3, F5, F7`.
   This is the set of correctors used to select perturbation files, not the
   operating-point layout. Every base and perturbed state still needs all 24
   vertical current readbacks in `cy_l` order.

2. **The C6 horizontal BPM is mislabeled C8 in the source files.** The notebook
   explains that its sixth horizontal BPM data channel named `C8` is physically
   C6. The API uses the physical name `PUEHC6` and expects BPM columns in
   `H_BPM_NAMES..., V_BPM_NAMES...` order. The notebook adapter must place the
   raw `horPosC8M` values in the API's C6 column. There is no separate C8
   column in the API's 33-BPM layout.

3. **The notebook likelihood is based on orbit responses.** It models
   `orbit_perturbed - orbit_base` and uses `sqrt(2) * measurement_error`. The
   current API likelihood compares an absolute predicted orbit with each row of
   `observed_orbit_mm`. Passing the notebook's response matrix as though it
   contained absolute orbits changes the statistical model. Reproducing the
   notebook requires either paired base/perturbed predictions and a response
   likelihood, or the original absolute BPM data for every state.

4. **Other machine inputs default to zero or the built-in operating point.**
   The notebook provides the five static currents and corrector readbacks, but
   does not provide `bdot`, trim/stopband currents, AC quadrupole current, or
   kicker settings. Confirm that their API defaults describe the measurements
   before treating the two models as equivalent.

The notebook's `trans-var-x-1:24` followed by `trans-var-y-1:24` is consistent
with the API's quadrupole-factor order only if the surrogate was trained with
horizontal quadrupoles in `QHA2, QHA4, …, QHF8` order and vertical quadrupoles
in `QVA1, QVA3, …, QVF7` order.

## 3. Backend-neutral operations

These are the preferred operation names:

| Operation | Single factor vector | Several factor vectors |
|---|---|---|
| Construct workspace | `batch_problem(prepared, backend; ...)` | `batch_problem(prepared, nfactors, backend; ...)` |
| Likelihood and gradient | `value_and_gradient!(problem, factors)` | `factor_batch_value_and_gradient!(problem, factors)` |
| AD primitive | `booster_loglikelihood(problem, factors)` | `booster_factor_batch_loglikelihoods(problem, factors)` |

The same operation names dispatch to CPU or CUDA methods based on the problem
and input array types. Names such as `cuda_value_and_gradient!` remain as exact
aliases for existing callers; new code should use the generic names.

Every problem is mutable and serially reusable. It stores warm-started closed
orbits and scratch arrays. Never call one problem concurrently. Create one
problem per concurrent chain or worker.

## 4. CPU usage

### 4.1 One factor vector

```julia
using KernelAbstractions

sensitivity = BatchedForwardSensitivity(
    chunksize=8,
    abstol=1e-12,
    maxiter=30,
    warm_start=true,
    threaded=false,
)

problem = batch_problem(
    prepared,
    KernelAbstractions.CPU();
    sensitivity,
)

factors = ones(48)
loglikelihood, gradient = value_and_gradient!(problem, factors)
predicted_orbit_mm = predict_orbits(problem, factors)
```

The direct constructor is equivalent:

```julia
problem = BoosterBatchProblem(prepared; sensitivity)
```

`value_and_gradient!` returns a scalar `Float64` likelihood and a 48-element
gradient. `predict_orbits` returns an `(nexperiments, 33)` matrix.

### 4.2 Several factor vectors in one call

```julia
nfactors = 32
problem = batch_problem(
    prepared,
    nfactors,
    KernelAbstractions.CPU();
    sensitivity,
)

factors = ones(nfactors, 48)
loglikelihoods, gradients = factor_batch_value_and_gradient!(problem, factors)

@assert size(loglikelihoods) == (nfactors,)
@assert size(gradients) == (nfactors, 48)
```

The workspace size is fixed at construction. Construct a new problem when
`nfactors` changes.

Factor batching is useful when several factor vectors are available together.
Independent MCMC chains usually progress asynchronously, so persistent
single-factor workspaces are generally simpler for them.

### 4.3 Turing

The supported CPU model wrapper is:

```julia
prior = lognormal_quad_prior(0.03; center=:median)
model = booster_model(problem, prior)
```

`booster_loglikelihood` has ChainRules and ReverseDiff bridges. Reverse AD uses
the supplied implicit gradient and does not enter tracking or differentiate the
Newton iterations.

### 4.4 CPU concurrency recommendations

The Perlmutter benchmark favored persistent single-thread Julia processes over
one large multithreaded Julia process:

- Use one single-thread Julia process per physical core for sustained,
  independent chains.
- Pin processes to physical cores and keep `OPENBLAS_NUM_THREADS=1`.
- Disable SMT. Queue extra chains behind the physical-core workers.
- Use only as many processes as there are active chains.
- For at most 64 active chains on a dual-socket Perlmutter CPU node, prefer one
  socket. Use both sockets when the workload needs more physical cores.
- Keep a process alive for many evaluations so Julia startup and compilation
  are amortized.

A typical Slurm shape is:

```sh
srun --ntasks=64 --cpus-per-task=1 --cpu-bind=cores \
    env JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    julia --startup-file=no --project=/path/to/environment run_chain.jl
```

Inside each rank, construct one `BoosterBatchProblem` and reuse it for the
entire chain. `chain_problems` allocates several workspaces inside one Julia
process, but the current large-thread-count benchmark shows shared-runtime
scaling limitations. It remains useful at modest concurrency or when process
startup and memory dominate.

Keep `BatchedForwardSensitivity.threaded=false` when chains are parallelized
outside the tracker. Setting it to `true` asks BeamTracking to parallelize a
single experiment batch and can oversubscribe cores when combined with chain
threads or processes.

CPU chunk width is a tuning parameter. Width 4 gives a controlled CUDA
comparison but requires 12 derivative passes for 48 parameters. Test widths 8,
12, and 16 for the production CPU shape. Wider widths reduce passes while
increasing per-value storage and cache pressure.

## 5. CUDA usage

Load the CUDA frontend by itself. It contains its corresponding CPU module as
`BoosterBatchedUQCUDA.UQ`, which should be used to prepare data for its CUDA
constructors:

```julia
using CUDA
CUDA.functional() || error("CUDA is unavailable")
CUDA.allowscalar(false)

include("cuda/booster_batched_uq_cuda.jl")
using .BoosterBatchedUQCUDA
const UQ = BoosterBatchedUQCUDA.UQ

prepared = UQ.prepare_booster_batch(
    operating_points,
    observed_orbit_mm;
    noise_std_mm=0.2,
)
```

Avoid separately including `booster_batched_uq.jl` first in the same script.
Until this source tree is packaged as a Julia package extension, doing so
creates a second module instance with distinct Julia types.

### 5.1 Device-input API

Device inputs select the device-result methods:

```julia
single_problem = batch_problem(
    prepared,
    CUDA.CUDABackend();
    chunksize=4,
    device_eltype=Float64,
    warm_start=true,
    abstol=1e-12,
    maxiter=40,
)

factors_gpu = CuArray(ones(Float64, 48))
values_gpu, gradients_gpu = value_and_gradient!(single_problem, factors_gpu)
status = device_status(single_problem)

@assert size(values_gpu) == (1,)
@assert size(gradients_gpu) == (1, 48)
```

For several factor vectors:

```julia
nfactors = 128
factor_problem = batch_problem(
    prepared,
    nfactors,
    CUDA.CUDABackend();
    chunksize=4,
    device_eltype=Float64,
    warm_start=true,
    maxiter=40,
)

factors_gpu = CuArray(ones(Float64, nfactors, 48))
values_gpu, gradients_gpu = factor_batch_value_and_gradient!(
    factor_problem, factors_gpu
)
```

The inputs, likelihoods, gradients, orbit, `I-R`, tangent states, and
reductions stay in device arrays. Julia still launches the element kernels and
SciBmad still makes host-visible convergence decisions.

Device-input calls leave status checking to the caller. Check status before
using a result from an untrusted proposal:

```julia
status = device_status(problem)
all(Array(status.converged)) || error("closed orbit did not converge")
any(Array(status.orbit_failed)) && error("closed-orbit solve failed")
any(Array(status.sensitivity_failed)) && error("implicit solve failed")
any(Array(status.input_invalid)) && error("non-finite factor")
any(Array(status.tracking_lost)) && error("particle lost")
```

Status and result buffers belong to the workspace and are overwritten on its
next evaluation. Copy them if they must outlive that call.

### 5.2 Host-adapter API

A host vector or matrix selects a compatibility method that stages factors to
CUDA, checks device status, synchronizes, and copies results back:

```julia
value, gradient = value_and_gradient!(single_problem, ones(48))
```

For a single-factor problem, this returns `Float64` and `Vector{Float64}`. For
a factor problem it returns a host vector and matrix. This path is convenient
for an existing CPU-oriented sampler, but it places a host/device boundary on
every evaluation.

CUDA also exposes a likelihood-only operation:

```julia
values_gpu = loglikelihood_value!(factor_problem, factors_gpu)
```

It omits parameter sensitivities and is useful for finite differences,
screening proposals, and likelihood-only benchmarks.

### 5.3 CUDA precision and chunk width

Use `Float64` with `chunksize=4` on the tested A100 path:

- Float64 uses `SciBmad.find_closed_orbit` on the CUDA arrays.
- Width 4 passed the full CUDA validation through 32K flattened lanes.
- Float64 width 8 exceeded a kernel launch resource limit on the A100 even
  though global device memory was mostly free.
- Float32 remains an experimental performance path. It uses the custom device
  orbit solver and has shown materially larger gradient error.

Changing `device_eltype` or `chunksize` creates different compiled kernels.
Warm each exact `(precision, chunk width, batch shape)` before measuring it.

### 5.4 When CUDA is appropriate

CUDA is most appropriate when:

- many factor vectors are available together;
- `nexperiments × nfactors` is large enough to occupy the GPU;
- factors are already on the GPU or downstream work consumes device results;
- a persistent problem can amortize compilation and allocation; and
- throughput matters more than the latency of one chain.

Use CPU for a single chain or a small number of independent chains. The current
CUDA tracker is launch-bound: tracking crosses a host function boundary at
each lattice element, and the Nsight trace shows many short kernels, device
allocations, synchronizations, and one-byte device-to-host reductions. On the
tested A100, increasing the flattened batch from 1 to 32K lanes raised gradient
time by less than a factor of two, so CUDA provides useful batch throughput,
but its single-call latency remains high.

For several physical GPUs, launch one Julia process per GPU and bind each
process with `CUDA_VISIBLE_DEVICES`. Partition independent factor batches or
chains between those processes. A problem belongs to one device and must not
be shared concurrently.

## 6. Warm starts, reproducibility, and failure handling

Warm starts are enabled by default. The next closed-orbit solve starts from the
previous successful orbit in that workspace. This is appropriate for a chain
whose proposals move locally and is another reason each chain needs its own
problem.

Set `warm_start=false` when:

- comparing backends from the same zero initial guess;
- testing convergence independently of evaluation order; or
- proposals may jump between disconnected operating regions.

Use a larger `maxiter` only after checking whether a proposal is physically
reasonable. Tightening `abstol` below the arithmetic precision does not improve
the answer. CUDA Float32 enforces a practical tolerance floor.

## 7. Validation and benchmarks

Before using a new NVIDIA system:

```sh
julia --project=/path/to/julia/project tests/test_batched_uq_cuda.jl
```

For end-to-end Float64 width-4 CUDA inference:

```sh
bash benchmarks/run_cuda_inference_4gpu.sh \
    /path/to/julia/project cuda_inference_logs
```

For CPU placement and execution-mode comparisons:

```sh
bash run_cpu_inference_perlmutter.sh /path/to/julia/project cpu_inference_logs
```

The full CPU runner covers socket-local and whole-node configurations and can
exceed a one-hour allocation. Run selected modes or separate jobs when the
goal is production tuning rather than an exhaustive topology study.
