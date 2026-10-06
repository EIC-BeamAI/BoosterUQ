# Perlmutter CPU inference benchmark

Run this inside a NERSC Perlmutter exclusive full CPU-node allocation:

```sh
bash run_cpu_inference_perlmutter.sh ~/.julia/env/lean cpu_inference_logs
```

The allocation must expose all 128 physical cores and 256 SMT hardware threads;
do not request the allocation with `--hint=nomultithread`. The runner executes
eight configurations sequentially:

| Configuration | Execution model | Workers | Slurm placement |
|---|---|---:|---|
| `threads_socket_physical` | one Julia process | 64 | socket 0 physical cores |
| `threads_socket_smt` | one Julia process | 128 | socket 0, both threads per core |
| `processes_socket_physical` | one-thread Julia processes | 64 | socket 0 physical cores |
| `processes_socket_smt` | one-thread Julia processes | 128 | socket 0, both threads per core |
| `threads_node_physical` | one Julia process | 128 | both sockets, physical cores |
| `threads_node_smt` | one Julia process | 256 | both sockets, both threads per core |
| `processes_node_physical` | one-thread Julia processes | 128 | both sockets, physical cores |
| `processes_node_smt` | one-thread Julia processes | 256 | both sockets, both threads per core |

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

The eight modes are intentionally run in one allocation. This keeps the Julia
environment, node, NUMA topology, and system load comparable. `OPENBLAS_NUM_THREADS`
is fixed at one. Single-socket threaded runs are explicitly restricted to CPUs
`0-63` or `0-63,128-191` and memory nodes `0-3`; whole-node runs use CPUs
`0-127` or `0-255` and memory nodes `0-7`. Process runs use Slurm `map_cpu` and
`map_mem` lists with the same topology. This distinguishes within-socket
scaling, SMT effects, and the additional effect of crossing sockets.
