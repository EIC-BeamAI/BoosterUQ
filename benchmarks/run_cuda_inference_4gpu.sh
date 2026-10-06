#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 /path/to/julia/project [output-directory]" >&2
    exit 2
fi

project=$1
output=${2:-cuda_inference_benchmark_logs}
mkdir -p "$output"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
groups=(
    '1:1,1:64,8:8,128:256'
    '64:1,256:128'
    '1:1024,32:32,64:128'
    '1024:1,128:64'
)

nvidia-smi > "$output/nvidia-smi.log" 2>&1
julia --version > "$output/julia-version.log" 2>&1
: > "$output/exit-codes.txt"
pids=()
for gpu in 0 1 2 3; do
    log="$output/gpu${gpu}.log"
    echo "GPU $gpu: ${groups[$gpu]} -> $log"
    CUDA_VISIBLE_DEVICES="$gpu" JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        BOOSTER_INFERENCE_PAIRS="${groups[$gpu]}" \
        BOOSTER_INFERENCE_OUTPUT="$output/gpu${gpu}.csv" \
        julia --startup-file=no --project="$project" \
        "$script_dir/benchmark_cuda_inference.jl" > "$log" 2>&1 &
    pids+=("$!")
done

status=0
for gpu in 0 1 2 3; do
    if wait "${pids[$gpu]}"; then result=0; else result=$?; fi
    printf 'gpu=%s exit=%s\n' "$gpu" "$result" | tee -a "$output/exit-codes.txt"
    (( result == 0 )) || status=1
done

summary="$output/inference_float64_width4.csv"
head -n 1 "$output/gpu0.csv" > "$summary"
for gpu in 0 1 2 3; do
    tail -n +2 "$output/gpu${gpu}.csv" >> "$summary"
done
echo "Combined results: $summary"
exit "$status"
