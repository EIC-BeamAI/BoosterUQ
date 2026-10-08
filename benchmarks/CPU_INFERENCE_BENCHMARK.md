# Perlmutter CPU inference benchmark

Run this inside a NERSC Perlmutter exclusive full CPU-node allocation:

```sh
bash benchmarks/run_cpu_inference_perlmutter.sh \
    ~/.julia/env/lean cpu_inference_logs
```

The allocation must expose all 128 physical cores. Requesting the node with
`--hint=nomultithread` is appropriate. The runner executes four configurations
sequentially:

| Configuration | Execution model | Workers | Slurm placement |
|---|---|---:|---|
| `threads_socket` | one Julia process | 64 | socket 0 physical cores |
| `processes_socket` | one-thread Julia processes | 64 | socket 0 physical cores |
| `threads_node` | one Julia process | 128 | both sockets, physical cores |
| `processes_node` | one-thread Julia processes | 128 | both sockets, physical cores |

Every chain owns a persistent `BoosterBatchProblem`; no workspace is shared by
concurrent chains. BeamTracking's internal CPU multithreading is disabled, so
parallelism is only across independent inference chains. When the requested
chain count exceeds the worker count, each worker evaluates multiple chains
sequentially. The concurrent-process runs use Slurm task binding and a barrier
outside each timed sample; the reported time is the slowest active process,
which is the batch makespan.

The workload uses the same 11 `(experiments, chains)` combinations as the CUDA
benchmark, from `1 × 1` through `128 × 256` and `256 × 128`, with at most 32,768
experiment-chain points. Float64 and ForwardDiff width 4 are fixed. Three
post-warmup samples time both the complete likelihood-and-gradient evaluation
and likelihood alone. Input construction, workspace creation, compilation,
nominal-orbit initialization, result inspection, and process barriers are
outside the timed region. The CPU and Float64 CUDA workflows now both use
`SciBmad.find_closed_orbit`; the returned `I-R` is reused by the subsequent
implicit parameter-sensitivity solve.

Outputs include one CSV and log per configuration, `lscpu.log`, launch
metadata, exit codes, per-rank process CSVs, and the combined
`cpu_inference_float64_width4.csv`. The process rank files are useful for load
imbalance analysis. Compare `seconds` and derive chains/second and
points/second at equal `(experiments, chains)` values. Setup and warmup rows
must be excluded from timing summaries.

The four modes are intentionally run in one allocation. This keeps the Julia
environment, node, NUMA topology, and system load comparable.
`OPENBLAS_NUM_THREADS` is fixed at one. Slurm block placement and core binding
put the 64-worker modes on one socket and the 128-worker modes across both
sockets. `--hint=nomultithread` and one task per core prevent sibling-thread
placement. Process memory is bound locally to each task. This distinguishes
within-socket scaling from the additional effect of crossing sockets without
assuming that raw Linux CPU IDs belong to a particular job-step cpuset.

SMT modes were removed because every tested combination reduced throughput.
These independent chains already keep each physical core busy, so sibling
threads contend for execution resources, cache, and memory bandwidth without
adding useful parallel capacity.

## Physical-core results

All four modes completed on one EPYC 7763 node. Every timed row reported
`status=ok`. The table gives the median of three post-warmup complete
likelihood-and-gradient evaluations, in seconds.

| Experiments | Chains | Points | Threads, socket | Processes, socket | Threads, node | Processes, node |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 0.332 | 0.360 | 0.338 | 0.350 |
| 1 | 64 | 64 | 1.388 | 0.846 | 1.931 | 0.866 |
| 8 | 8 | 64 | 0.492 | 0.436 | 0.559 | 0.430 |
| 64 | 1 | 64 | 0.778 | 0.782 | 0.823 | 0.780 |
| 1 | 1,024 | 1,024 | 21.529 | 6.800 | 23.401 | 3.461 |
| 32 | 32 | 1,024 | 1.607 | 0.684 | 2.211 | 0.632 |
| 1,024 | 1 | 1,024 | 7.280 | 7.293 | 7.993 | 7.203 |
| 64 | 128 | 8,192 | 7.088 | 1.883 | 7.915 | 1.015 |
| 128 | 64 | 8,192 | 5.614 | 1.455 | 8.185 | 1.454 |
| 128 | 256 | 32,768 | 23.564 | 5.824 | 34.032 | 3.044 |
| 256 | 128 | 32,768 | 12.992 | 5.096 | 15.086 | 2.555 |

The results support three conclusions:

1. **Independent processes are the production mode for concurrent chains.**
   On one socket they are 2.35–4.05 times faster than Julia threads for the
   larger multi-chain workloads. Across the full node they are 3.50–11.18
   times faster. At one active chain the modes are effectively equal because
   the implementation parallelizes across chains, not across experiments in
   one chain.
2. **The process mode scales cleanly over the second socket.** With at least
   128 active chains, `processes_node` is 1.85–2.00 times faster than
   `processes_socket`. With 64 or fewer active chains, the second socket adds
   no useful capacity and the two process modes are within 3% except for one
   noisy 32-chain case.
3. **A single Julia process should stay within one socket.** `threads_node` is
   9–46% slower than `threads_socket` on the substantial workloads. Its
   shared runtime, heap, and cross-socket memory traffic outweigh the extra
   cores for this chain-level task structure.

Likelihood-only timings show the same trend once each chain has enough work.
For `128 × 256`, the four modes took 4.233, 0.667, 6.850, and 0.462 seconds in
the table's column order. Very small process workloads are sensitive to the
slowest rank: at `1 × 64`, the slowest rank took about twice the median rank
time. The aggregate deliberately reports that maximum because it is the real
batch completion time. On the large process workloads, the slowest rank was
within 2–12% of the median.

## Usage recommendations

- Use one single-thread Julia process per physical core and one independent
  chain per process. Use `processes_node` when at least 128 chains can be kept
  active, and `processes_socket` for 64 or fewer concurrent chains or when a
  socket is the available allocation unit.
- Match the number of launched processes to the number of concurrent chains
  in production. The benchmark always launches 64 or 128 ranks for controlled
  comparisons, even when most are inactive.
- For one chain, use one core. Increasing the worker allocation does not split
  that chain's experiment batch across workers.
- Prefer more experiments per chain when the statistical workflow permits it.
  At 32,768 total points, `256 × 128` is 1.19 times faster than `128 × 256` in
  `processes_node`, showing that larger per-chain batches amortize inference
  overhead better.
- Retain `threads_socket` as a convenient in-process mode for small local
  workloads. Avoid `threads_node` for production on this architecture.

The directly addressable bottleneck was NUMA and runtime interference between
independent chains in one Julia process; process isolation with local memory
binding resolves it. The remaining single-chain time is the serial inference
cost inside one `BoosterBatchProblem`. Reducing that requires optimizing the
tracking, closed-orbit, and AD kernels themselves or assigning multiple factor
sets to one batched device-style problem. Adding CPU workers cannot reduce it
under the current one-chain-per-worker design.
