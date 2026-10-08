#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 /path/to/julia/project [output-directory]" >&2
    exit 2
fi
command -v srun >/dev/null || { echo "srun is unavailable" >&2; exit 2; }

project=$1
output=${2:-cpu_inference_benchmark_logs}
mkdir -p "$output"
output=$(cd "$output" && pwd)
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
scratch_root=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
[[ -d "$scratch_root" ]] || scratch_root=/tmp
pairs=${BOOSTER_CPU_PAIRS:-'1:1,1:64,8:8,64:1,1:1024,32:32,1024:1,64:128,128:64,128:256,256:128'}
samples=${BOOSTER_CPU_SAMPLES:-3}
status=0

lscpu > "$output/lscpu.log" 2>&1
julia --version > "$output/julia-version.log" 2>&1
{
    date
    printf 'host=%s\n' "$(hostname)"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-unset}"
    printf 'pairs=%s\n' "$pairs"
    printf 'samples=%s\n' "$samples"
} > "$output/metadata.log"
: > "$output/exit-codes.txt"

run_threads() {
    local config=$1 workers=$2 sockets=$3
    echo "Running $config: one chain per Julia thread ($workers workers)"
    if srun --nodes=1 --ntasks=1 --cpus-per-task="$workers" \
        --sockets-per-node="$sockets" --cores-per-socket=64 \
        --threads-per-core=1 --hint=nomultithread \
        --distribution=block:block --cpu-bind=cores \
        --exclusive --exact \
        env JULIA_NUM_THREADS="$workers" JULIA_EXCLUSIVE=1 OPENBLAS_NUM_THREADS=1 \
        BOOSTER_CPU_CONFIG="$config" BOOSTER_CPU_PAIRS="$pairs" \
        BOOSTER_CPU_SAMPLES="$samples" BOOSTER_CPU_CHUNK=4 \
        BOOSTER_CPU_OUTPUT="$output/$config.csv" \
        julia --startup-file=no --project="$project" \
        "$script_dir/benchmark_cpu_inference.jl" > "$output/$config.log" 2>&1
    then result=0; else result=$?; status=1; fi
    printf '%s exit=%s\n' "$config" "$result" | tee -a "$output/exit-codes.txt"
}

run_processes() {
    local config=$1 workers=$2 sockets=$3 barrier
    barrier=$(mktemp -d "$scratch_root/${config}-barriers.XXXXXX") || return 2
    echo "Running $config: $workers concurrent single-thread Julia processes"
    if srun --nodes=1 --ntasks="$workers" --cpus-per-task=1 \
        --ntasks-per-core=1 --ntasks-per-socket=64 \
        --sockets-per-node="$sockets" --cores-per-socket=64 \
        --threads-per-core=1 --hint=nomultithread \
        --distribution=block:block --cpu-bind=cores --mem-bind=local \
        --kill-on-bad-exit=1 --exclusive --exact \
        env JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        BOOSTER_CPU_PROCESS_MODE=1 BOOSTER_CPU_CONFIG="$config" \
        BOOSTER_CPU_PAIRS="$pairs" BOOSTER_CPU_SAMPLES="$samples" \
        BOOSTER_CPU_CHUNK=4 BOOSTER_CPU_BARRIER_DIRECTORY="$barrier" \
        BOOSTER_CPU_OUTPUT="$output/$config.csv" \
        julia --startup-file=no --project="$project" \
        "$script_dir/benchmark_cpu_inference.jl" > "$output/$config.log" 2>&1
    then result=0; else result=$?; status=1; fi
    printf '%s exit=%s\n' "$config" "$result" | tee -a "$output/exit-codes.txt"
}

# Block placement fills one socket before crossing to the second socket.
run_threads threads_socket 64 1
run_processes processes_socket 64 1
run_threads threads_node 128 2
run_processes processes_node 128 2

summary="$output/cpu_inference_float64_width4.csv"
first=1
for config in \
    threads_socket processes_socket \
    threads_node processes_node
do
    file="$output/$config.csv"
    [[ -f "$file" ]] || continue
    if (( first )); then
        head -n 1 "$file" > "$summary"
        first=0
    fi
    tail -n +2 "$file" >> "$summary"
done
(( first )) || echo "Combined results: $summary"
exit "$status"
