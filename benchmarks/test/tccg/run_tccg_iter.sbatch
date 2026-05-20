#!/bin/bash
#SBATCH -J TCCG_ALL
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH -o all.out
#SBATCH -e %x.%j.err
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --comment etc

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

find_repo_root() {
    local dir="$1"
    while [[ "$dir" != "/" ]]; do
        if [[ -f "$dir/CMakeLists.txt" ]]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

REPO_ROOT="${REPO_ROOT:-$(find_repo_root "${SCRIPT_DIR}")}"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
TARGET="${TARGET:-bench_tccg_iter}"
BINARY="${BINARY:-${BUILD_DIR}/benchmarks/test/tccg/bench_tccg_iter}"
BENCHMARK_ID="${BENCHMARK_ID:-1}"
BACKEND="${BACKEND:-ttgt}"
WARMUP="${WARMUP:-10}"
REPEATS="${REPEATS:-200}"
PRECISION_FLAG="${PRECISION_FLAG:---fp64}"
VERIFY_FLAG="${VERIFY_FLAG:-}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results/job_${SLURM_JOB_ID:-local}}"
OUTPUT_CSV="${OUTPUT_CSV:-${RESULT_DIR}/eq${BENCHMARK_ID}_${BACKEND}_iter.csv}"
COGENT_CUBIN_COMPILE_OPT="${COGENT_CUBIN_COMPILE_OPT:-DEFAULT}"

module purge
module load gcc/11.5.0 cuda/12.9.1

mkdir -p "${RESULT_DIR}"

echo "[Iter] Repo root  : ${REPO_ROOT}"
echo "[Iter] Build dir  : ${BUILD_DIR}"
echo "[Iter] Target     : ${TARGET}"
echo "[Iter] Binary     : ${BINARY}"
echo "[Iter] Benchmark  : ${BENCHMARK_ID}"
echo "[Iter] Backend    : ${BACKEND}"
echo "[Iter] Warmup     : ${WARMUP}"
echo "[Iter] Repeats    : ${REPEATS}"
echo "[Iter] Precision  : ${PRECISION_FLAG}"
echo "[Iter] Verify     : ${VERIFY_FLAG:-<off>}"
echo "[Iter] Output CSV : ${OUTPUT_CSV}"

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -DCOGENT_CUBIN_COMPILE_OPT="${COGENT_CUBIN_COMPILE_OPT}"
cmake --build "${BUILD_DIR}" --target "${TARGET}" -j8

"${BINARY}" \
    -b "${BENCHMARK_ID}" \
    --backend "${BACKEND}" \
    --warmup "${WARMUP}" \
    --repeats "${REPEATS}" \
    "${PRECISION_FLAG}" \
    ${VERIFY_FLAG} \
    --output "${OUTPUT_CSV}"
