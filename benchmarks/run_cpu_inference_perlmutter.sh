#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 /path/to/julia/project [output-directory]" >&2
    exit 2
fi
command -v srun >/dev/null || { echo "srun is unavailable" >&2; exit 2; }
command -v numactl >/dev/null || { echo "numactl is unavailable" >&2; exit 2; }

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

range_csv() {
    local first=$1 last=$2 result= value
    for ((value=first; value<=last; value++)); do
        result+="${result:+,}$value"
    done
    printf '%s' "$result"
}

socket_physical=$(range_csv 0 63)
socket_smt="$socket_physical,$(range_csv 128 191)"
node_physical=$(range_csv 0 127)
node_smt=$(range_csv 0 255)

memory_map() {
    local cpu_list=$1 result= cpu core domain
    IFS=',' read -r -a cpus <<< "$cpu_list"
    for cpu in "${cpus[@]}"; do
        core=$(( cpu % 128 ))
        domain=$(( core / 16 ))
        result+="${result:+,}$domain"
    done
    printf '%s' "$result"
}

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
    local config=$1 workers=$2 cpu_list=$3 memory_nodes=$4
    echo "Running $config: one chain per Julia thread ($workers workers)"
    # Give the step the full node, then constrain both execution and allocation
    # to the requested socket(s). Julia pins its threads within this mask.
    if srun --nodes=1 --ntasks=1 --cpus-per-task=256 \
        --cpu-bind=none --exclusive --exact \
        numactl --physcpubind="$cpu_list" --membind="$memory_nodes" \
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
    local config=$1 workers=$2 cpu_list=$3 barrier memory_list
    barrier=$(mktemp -d "$scratch_root/${config}-barriers.XXXXXX") || return 2
    memory_list=$(memory_map "$cpu_list")
    echo "Running $config: $workers concurrent single-thread Julia processes"
    if srun --nodes=1 --ntasks="$workers" --cpus-per-task=1 \
        --cpu-bind="map_cpu:$cpu_list" --mem-bind="map_mem:$memory_list" \
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

# Socket 0: cores 0-63, siblings 128-191, NUMA nodes 0-3.
# Whole node: cores 0-127, siblings 128-255, NUMA nodes 0-7.
run_threads threads_socket_physical 64 "$socket_physical" 0-3
run_threads threads_socket_smt 128 "$socket_smt" 0-3
run_processes processes_socket_physical 64 "$socket_physical"
run_processes processes_socket_smt 128 "$socket_smt"
run_threads threads_node_physical 128 "$node_physical" 0-7
run_threads threads_node_smt 256 "$node_smt" 0-7
run_processes processes_node_physical 128 "$node_physical"
run_processes processes_node_smt 256 "$node_smt"

summary="$output/cpu_inference_float64_width4.csv"
first=1
for config in \
    threads_socket_physical threads_socket_smt \
    processes_socket_physical processes_socket_smt \
    threads_node_physical threads_node_smt \
    processes_node_physical processes_node_smt
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
