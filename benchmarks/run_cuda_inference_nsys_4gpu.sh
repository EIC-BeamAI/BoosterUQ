#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 /path/to/julia/project [output-directory]" >&2
    exit 2
fi
command -v nsys >/dev/null || { echo "nsys is unavailable; load cudatoolkit first" >&2; exit 2; }
nsys_root=${NSYS_ROOT:-$(dirname "$(dirname "$(readlink -f "$(command -v nsys)")")")}
importer="$nsys_root/host-linux-x64/QdstrmImporter"
[[ -x "$importer" ]] || {
    echo "QdstrmImporter is unavailable: $importer (set NSYS_ROOT to the Nsight_Systems directory)" >&2
    exit 2
}

project=$1
output=${2:-cuda_inference_nsys_logs}
mkdir -p "$output"
output=$(cd "$output" && pwd)
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
scratch_root=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
[[ -d "$scratch_root" ]] || scratch_root=/tmp
nsys_scratch=$(mktemp -d "$scratch_root/booster-nsys.XXXXXX") || exit 2
trap 'rm -rf "$nsys_scratch"' EXIT
groups=(
    '1:1,1:64,8:8,128:256'
    '64:1,256:128'
    '1:1024,32:32,64:128'
    '1024:1,128:64'
)
profile_pairs=('1:1' '256:128' '1:1024' '1024:1')

# Nsight's launcher can load the system libcrypto before Julia's bundled libssl.
# Put Julia's matching libcrypto into the target process before either initializes.
julia_libdir=$(julia --startup-file=no -e 'print(realpath(joinpath(Sys.BINDIR, "..", "lib", "julia")))') || exit 2
libcrypto="$julia_libdir/libcrypto.so.3"
[[ -f "$libcrypto" ]] || { echo "Julia libcrypto is missing: $libcrypto" >&2; exit 2; }
profile_library_path="$julia_libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
profile_preload="$libcrypto${LD_PRELOAD:+ $LD_PRELOAD}"
if ! LD_LIBRARY_PATH="$profile_library_path" LD_PRELOAD="$profile_preload" \
    julia --startup-file=no --project="$project" -e 'using OpenSSL_jll' \
    > "$output/openssl-check.log" 2>&1; then
    cat "$output/openssl-check.log" >&2
    exit 2
fi
LD_LIBRARY_PATH="$profile_library_path" LD_PRELOAD="$profile_preload" \
    ldd "$julia_libdir/libssl.so.3" \
    > "$output/openssl-loader.log" 2>&1 || true

nsys --version > "$output/nsys-version.log" 2>&1
nvidia-smi > "$output/nvidia-smi.log" 2>&1
: > "$output/exit-codes.txt"
pids=()
for gpu in 0 1 2 3; do
    echo "Benchmarking GPU $gpu: ${groups[$gpu]} (profiling ${profile_pairs[$gpu]})"
    CUDA_VISIBLE_DEVICES="$gpu" JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        NSYS_TMPDIR="$nsys_scratch" \
        LD_LIBRARY_PATH="$profile_library_path" LD_PRELOAD="$profile_preload" \
        BOOSTER_INFERENCE_PAIRS="${groups[$gpu]}" \
        BOOSTER_INFERENCE_PROFILE=1 \
        BOOSTER_INFERENCE_PROFILE_PAIR="${profile_pairs[$gpu]}" \
        BOOSTER_INFERENCE_OUTPUT="$output/gpu${gpu}.csv" \
        nsys profile --trace=cuda,nvtx --sample=none \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        --force-overwrite=true --output="$nsys_scratch/gpu${gpu}" \
        julia --startup-file=no --project="$project" \
        "$script_dir/benchmark_cuda_inference.jl" > "$output/gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done

status=0
for gpu in 0 1 2 3; do
    if wait "${pids[$gpu]}"; then result=0; else result=$?; fi
    printf 'gpu=%s exit=%s\n' "$gpu" "$result" | tee -a "$output/exit-codes.txt"
    (( result == 0 )) || status=1
done

summary="$output/inference_float64_width4.csv"
if [[ -f "$output/gpu0.csv" ]]; then
    head -n 1 "$output/gpu0.csv" > "$summary"
    for gpu in 0 1 2 3; do
        [[ -f "$output/gpu${gpu}.csv" ]] &&
            tail -n +2 "$output/gpu${gpu}.csv" >> "$summary"
    done
    echo "Combined timings: $summary"
fi

for gpu in 0 1 2 3; do
    report="$nsys_scratch/gpu${gpu}.nsys-rep"
    stream="$nsys_scratch/gpu${gpu}.qdstrm"
    if [[ ! -f "$report" && -f "$stream" ]]; then
        "$importer" --input-file="$stream" --output-file="$report" \
            --force-overwrite > "$output/gpu${gpu}-import.log" 2>&1 || true
    fi
    if [[ -f "$report" ]]; then
        nsys stats --report cuda_api_sum --report cuda_gpu_kern_sum \
            --report cuda_gpu_mem_time_sum "$report" \
            > "$output/gpu${gpu}-stats.log" 2>&1 || \
            echo "nsys stats failed for GPU $gpu; inspect $output/gpu${gpu}.nsys-rep" >&2
        cp "$report" "$output/gpu${gpu}.nsys-rep" || status=1
    else
        [[ -f "$stream" ]] && cp "$stream" "$output/gpu${gpu}.qdstrm"
        echo "Nsight conversion failed for GPU $gpu; inspect $output/gpu${gpu}-import.log" >&2
        status=1
    fi
done
echo "Nsight reports and logs: $output"
exit "$status"
