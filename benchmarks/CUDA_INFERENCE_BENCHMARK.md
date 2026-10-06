# CUDA Float64 width-4 inference benchmark

Run from `BoosterUQ` on the four-GPU machine:

```sh
bash run_cuda_inference_4gpu.sh /path/to/julia/project /path/to/inference-benchmark-logs
```

The project should be the same environment used for the passing CUDA validation. The runner launches one Julia process per GPU and covers 11 `(experiments, factor sets)` combinations from one to 32,768 flattened lanes. Each GPU gets a different group, including both `(128,256)` and `(256,128)` at 32K lanes. One factor set uses the single-vector `cuda_problem` dispatch; larger factor batches use `cuda_factor_problem`.

Each combination constructs a persistent CUDA workspace and warms both calls. Three subsequent samples time each mode:

- `value_gradient`: complete `value_and_gradient!`, including device factor preparation, `SciBmad.find_closed_orbit`, BPM tracking, likelihood reduction, parameter AD, and the on-device implicit `I−R` response solve.
- `likelihood`: complete `loglikelihood_value!`, including the orbit and BPM prediction, without parameter gradients.

Inputs are `CuArray` factor sets before the timer starts. Outputs remain on the GPU. The timer synchronizes the GPU before and after each call. Problem construction, compilation, input allocation, convergence checks, and result inspection are outside the timed region. `setup` and warmup times are recorded separately. Warm starts are enabled; each timed sample perturbs the factors. The benchmark checks finite values and gradients, convergence, tracking loss, and singular solves after each call.

Results appear in `gpu0.csv` through `gpu3.csv` and the combined `inference_float64_width4.csv`, alongside per-GPU logs, device metadata, and `exit-codes.txt`. The script returns a nonzero exit code if any combination fails. Use `BOOSTER_INFERENCE_SAMPLES` to change the number of timed samples. For a single-GPU custom run, set `BOOSTER_INFERENCE_PAIRS` to comma-separated `experiments:factors` pairs and run `benchmark_cuda_inference.jl` directly. These timings cover one inference evaluation, rather than an entire sampling chain.

## Nsight Systems

After loading Perlmutter's `cudatoolkit` module so that `nsys` is on `PATH`, run:

```sh
bash run_cuda_inference_nsys_4gpu.sh /path/to/julia/project /path/to/inference-nsys-logs
```

This single runner covers the same 11 combinations and both timing modes as the regular runner. It also captures one complete `value_and_gradient!` call per GPU, at `(experiments, factor sets)` of `(1,1)`, `(256,128)`, `(1,1024)`, and `(1024,1)`. Each process warms its workspace before that capture; all regular timed samples are outside the capture range. The four reports are `gpu0.nsys-rep` through `gpu3.nsys-rep`; `gpu*-stats.log` summarizes CUDA API calls, kernels, and memory operations. The combined timing table is `inference_float64_width4.csv`; `value_gradient_profile` rows identify the extra captured evaluations and should be excluded from timing summaries.

The runner puts Julia's bundled `libcrypto.so.3` ahead of the system copy when launching Nsight. This addresses the Perlmutter loader failure in which `/usr/lib64/libcrypto.so.3` lacks the `OPENSSL_3.3.0` symbol required by Julia's `libssl.so.3`. It checks `using OpenSSL_jll` before launching the four jobs and records the dynamic-loader resolution in `openssl-loader.log`. Nsight now writes and imports reports in node-local scratch, then copies completed `.nsys-rep` files to the requested output directory.

The October 7 capture completed all timings, but Nsight's importer failed to write reports directly to the shared `/global` path (`CreateFileException`, errno 524). The four `.qdstrm` files are valid intermediate capture streams. Recover these existing captures without rerunning the benchmark, using the same Nsight Systems version (`2026.2.1.210-262137639646v0`):

```sh
module load cudatoolkit
bash recover_cuda_inference_nsys.sh /path/to/existing/cuda_val
```

The recovery script uses Nsight's `host-linux-x64/QdstrmImporter` in node-local scratch, copies the report into the log directory, and writes `gpu*-stats.log`. The `.qdstrm` files remain untouched. The loaded `nsys` installation is used to locate the importer; set `NSYS_ROOT` to the `Nsight_Systems` directory if it is elsewhere. Perlmutter's `nsys` does not provide an `import` subcommand.

### Timing results

All four GPU processes exited successfully. The median of three unprofiled samples per configuration is:

| Experiments × factor sets | Lanes | Likelihood + gradient | Likelihood |
|---|---:|---:|---:|
| 1 × 1 | 1 | 4.18 s | 0.71 s |
| 1 × 64 | 64 | 4.84 s | 0.67 s |
| 64 × 1 | 64 | 4.55 s | 0.58 s |
| 8 × 8 | 64 | 4.57 s | 0.69 s |
| 1 × 1024 | 1024 | 4.53 s | 0.70 s |
| 1024 × 1 | 1024 | 5.09 s | 0.76 s |
| 32 × 32 | 1024 | 4.75 s | 0.71 s |
| 64 × 128 | 8192 | 5.83 s | 0.77 s |
| 128 × 64 | 8192 | 5.59 s | 0.78 s |
| 128 × 256 | 32768 | 6.96 s | 0.96 s |
| 256 × 128 | 32768 | 6.37 s | 1.05 s |

The gradient step takes roughly six to eight times as long as likelihood alone, while the 32K-lane runs are less than twice the one-lane time. These samples were outside the capture range but still ran under an `nsys`-launched process.

The recovered GPU 0 report captures one `1 × 1` gradient evaluation. Its CUDA API summary contains 210,461 kernel launches, 195,850 asynchronous device allocations, 44,446 stream synchronizations, and 15,267 device-to-host copies. The SQLite trace shows that 15,263 copies were one byte each, consistent with scalar Boolean reductions rather than bulk particle transfers. The report also shows 1,218,076 `cuStreamIsCapturing` calls. On-device compute is not the only cost: Julia repeatedly launches short kernels and waits for host-visible results. The custom `_row_solve!` appears only 12 times and accounts for about 0.21 ms of GPU kernel time, so optimizing that 4×4 solve cannot address the main cost.

Grouping the 210,500 traced GPU kernels by name gives 181,212 broadcast kernels (544 ms cumulative device time), 15,268 reduction kernels (63 ms), 13,905 BeamTracking generic element kernels (519 ms), and 115 others (less than 1 ms). These cumulative kernel times are GPU activity, not end-to-end elapsed time. The broadcast and allocation counts strongly implicate repeated `BatchParam` array arithmetic during element unpacking; the trace lacks Julia call stacks, so this is a source-level inference rather than a measured call-site attribution. `BeamTracking/src/kernel.jl` launches and synchronizes each generic element kernel, and the UQ gradient loop tracks the line twice for each of 12 parameter chunks.

A likely source of the one-byte copies is BeamTracking's `BatchParam` reference-momentum comparison inside element unpacking: `unpack_bl.jl` compares lane-dependent `BatchParam` values with `≈`, and `batch.jl` implements `isapprox(::BatchParam, ::BatchParam)` using `all` over the device array. That reduction returns a host Boolean at every such comparison. The trace lacks a Julia call stack, so the exact share of copies from this site remains to be measured. A safe fix belongs in BeamTracking's reference handling: skip the comparison when the incoming and bunch references are provably identical, and otherwise keep the reference shift decision correct for every lane. The UQ code should not replace `isapprox` with an unconditional true result.

Use the unprofiled `value_gradient` and `likelihood` rows for elapsed times; instrumentation adds overhead to the `value_gradient_profile` row. The Nsight trace is for locating kernel, launch, synchronization, and transfer costs. Running under `nsys` may still add some process-level overhead to unprofiled rows, so use the plain runner only if a baseline independent of Nsight is needed. The runner checks for `nsys` before launching, and a missing report or failed benchmark produces a nonzero exit status. If the installed Nsight version cannot produce one of the optional stats reports, the `.nsys-rep` remains the primary profiling artifact. [NVIDIA documents the CUDA profiler API capture range and Nsight statistics reports](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).
