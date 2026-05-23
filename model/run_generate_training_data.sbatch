#!/bin/bash
#SBATCH -J TCONT_MODEL_DATA
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --time=2-00:00:00
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
        if [[ -f "$dir/CMakeLists.txt" && -d "$dir/model" ]]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

REPO_ROOT="${REPO_ROOT:-}"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(find_repo_root "${SCRIPT_DIR}")" || {
        echo "[Error] Could not find the TconT repo root from ${SCRIPT_DIR}" >&2
        exit 1
    }
fi
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"

module purge
module load gcc/11.5.0 cuda/12.9.1 python/3.14.2 cmake/4.1.3

ROOT_BUILD_DIR="${ROOT_BUILD_DIR:-${REPO_ROOT}/build_model_root}"
MODEL_BUILD_DIR="${MODEL_BUILD_DIR:-${REPO_ROOT}/build_model_tools}"
COGENT_CUBIN_COMPILE_OPT="${COGENT_CUBIN_COMPILE_OPT:-A100}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-8}}"

INPUT_ROOT="${INPUT_ROOT:-${REPO_ROOT}/../gen_input/rand_problems_new2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/model/results}"
PRECISION="${PRECISION:-fp64}"
WARMUP="${WARMUP:-100}"
REPEATS="${REPEATS:-200}"
TIMEOUT_SEC="${TIMEOUT_SEC:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
LIMIT="${LIMIT:-0}"
APPEND_FLAG="${APPEND_FLAG:-0}"
VERBOSE_FLAG="${VERBOSE_FLAG:-1}"
BACKENDS="${BACKENDS:-cogent ttgt}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-${REPO_ROOT}/model/generate_training_data.py}"
BUILD_SCRIPT="${BUILD_SCRIPT:-${REPO_ROOT}/model/build_bench_rand_problem.sh}"

OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/$(basename "${INPUT_ROOT}")/shard_${SHARD_INDEX}_of_${NUM_SHARDS}}"

echo "=========================================="
echo "SLURM_JOB_ID     = ${SLURM_JOB_ID:-local}"
echo "SLURM_NODELIST   = ${SLURM_NODELIST:-local}"
echo "Repo root        = ${REPO_ROOT}"
echo "Root build dir   = ${ROOT_BUILD_DIR}"
echo "Model build dir  = ${MODEL_BUILD_DIR}"
echo "Input root       = ${INPUT_ROOT}"
echo "Output dir       = ${OUTPUT_DIR}"
echo "Precision        = ${PRECISION}"
echo "Warmup / repeats = ${WARMUP} / ${REPEATS}"
echo "Shard            = ${SHARD_INDEX}/${NUM_SHARDS}"
echo "Backends         = ${BACKENDS}"
echo "=========================================="

cd "${REPO_ROOT}"

bash "${BUILD_SCRIPT}" \
    --root-build-dir "${ROOT_BUILD_DIR}" \
    --model-build-dir "${MODEL_BUILD_DIR}" \
    --jobs "${JOBS}" \
    --cogent-cubin-compile-opt "${COGENT_CUBIN_COMPILE_OPT}"

BINARY_PATH="${MODEL_BUILD_DIR}/bench_rand_problem"
mkdir -p "${OUTPUT_DIR}"
export LD_LIBRARY_PATH="${ROOT_BUILD_DIR}/src:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

PY_ARGS=(
    --binary "${BINARY_PATH}"
    --input-root "${INPUT_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --precision "${PRECISION}"
    --warmup "${WARMUP}"
    --repeats "${REPEATS}"
    --num-shards "${NUM_SHARDS}"
    --shard-index "${SHARD_INDEX}"
    --timeout-sec "${TIMEOUT_SEC}"
    --backends ${BACKENDS}
)

if [[ "${LIMIT}" != "0" ]]; then
    PY_ARGS+=(--limit "${LIMIT}")
fi

if [[ "${APPEND_FLAG}" == "1" ]]; then
    PY_ARGS+=(--append)
fi

if [[ "${VERBOSE_FLAG}" == "1" ]]; then
    PY_ARGS+=(--verbose)
fi

python3 "${PYTHON_SCRIPT}" "${PY_ARGS[@]}"
