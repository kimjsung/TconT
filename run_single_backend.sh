#!/bin/bash
#SBATCH -J TCCG_ALL
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -o all.out
#SBATCH -e %x.%j.err
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=single:1
#SBATCH --comment etc

set -euo pipefail

# In a Slurm job, BASH_SOURCE[0] may point to Slurm's internal spool path
# such as /var/spool/slurm/..., where normal users cannot create files.
# Therefore, use SLURM_SUBMIT_DIR as the base directory inside Slurm.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# Find the repository root robustly by walking upward until CMakeLists.txt is found.
# This works whether sbatch is launched from the repo root or from a subdirectory.
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

REPO_ROOT="$(find_repo_root "${SCRIPT_DIR}")" || {
    echo "[Error] Could not find CMakeLists.txt by walking upward from SCRIPT_DIR=${SCRIPT_DIR}" >&2
    echo "[Error] Run sbatch from inside the TconT repository, or set REPO_ROOT explicitly." >&2
    exit 1
}

# Find the Python driver. Prefer the submit directory, then common repo locations.
PY_DRIVER="${PY_DRIVER:-}"
if [[ -z "${PY_DRIVER}" ]]; then
    if [[ -f "${SCRIPT_DIR}/run_tccg_benchmarks.py" ]]; then
        PY_DRIVER="${SCRIPT_DIR}/run_tccg_benchmarks.py"
    elif [[ -f "${REPO_ROOT}/run_tccg_benchmarks.py" ]]; then
        PY_DRIVER="${REPO_ROOT}/run_tccg_benchmarks.py"
    elif [[ -f "${REPO_ROOT}/benchmarks/tccg/run_tccg_benchmarks.py" ]]; then
        PY_DRIVER="${REPO_ROOT}/benchmarks/tccg/run_tccg_benchmarks.py"
    else
        echo "[Error] Could not find run_tccg_benchmarks.py" >&2
        echo "[Error] Looked in:" >&2
        echo "        ${SCRIPT_DIR}/run_tccg_benchmarks.py" >&2
        echo "        ${REPO_ROOT}/run_tccg_benchmarks.py" >&2
        echo "        ${REPO_ROOT}/benchmarks/tccg/run_tccg_benchmarks.py" >&2
        exit 1
    fi
fi

module purge
module load gcc/11.5.0 cuda/12.9.1 python/3.14.2

CMAKE_BUILD_DIR="${CMAKE_BUILD_DIR:-${REPO_ROOT}/build}"
CMAKE_TARGET="${CMAKE_TARGET:-bench_tccg}"
CMAKE_CONFIGURE_ARGS="${CMAKE_CONFIGURE_ARGS:-}"
CMAKE_BUILD_ARGS="${CMAKE_BUILD_ARGS:--j8}"
GPU_TARGET="${1:-${GPU_TARGET:-}}"
if [[ -n "${GPU_TARGET}" ]]; then
    case "${GPU_TARGET,,}" in
        a100)
            COGENT_CUBIN_COMPILE_OPT="A100"
            ;;
        h100)
            COGENT_CUBIN_COMPILE_OPT="H100"
            ;;
        h200)
            COGENT_CUBIN_COMPILE_OPT="H200"
            ;;
        *)
            echo "[Error] Unsupported GPU target argument: ${GPU_TARGET}" >&2
            echo "[Error] Usage: $0 [a100|h100|h200]" >&2
            exit 1
            ;;
    esac
else
    COGENT_CUBIN_COMPILE_OPT="${COGENT_CUBIN_COMPILE_OPT:-A100}"
fi
BINARY_PATH="${BINARY_PATH:-${CMAKE_BUILD_DIR}/benchmarks/tccg/bench_tccg}"
START_EQ="${START_EQ:-1}"
END_EQ="${END_EQ:-48}"
VERIFY_FLAG="${VERIFY_FLAG:---verify}"
BACKEND="${BACKEND:-model}"
RESULT_ROOT="${RESULT_ROOT:-${SCRIPT_DIR}/results}"
RESULT_DIR="${RESULT_DIR:-${RESULT_ROOT}/${BACKEND}/job_${SLURM_JOB_ID:-local}}"
PYTHON_ARGS=()
export PYTHONUNBUFFERED=1

case "${VERIFY_FLAG}" in
    ""|"--verify")
        ;;
    "--no-verify")
        PYTHON_ARGS+=("--no-verify")
        ;;
    *)
        echo "[Error] Unsupported VERIFY_FLAG: ${VERIFY_FLAG}" >&2
        echo "[Error] Use one of: '', --verify, --no-verify" >&2
        exit 1
        ;;
esac

case "${BACKEND}" in
    model|cogent|ttgt)
        ;;
    *)
        echo "[Error] Unsupported BACKEND: ${BACKEND}" >&2
        echo "[Error] Use one of: model, cogent, ttgt" >&2
        exit 1
        ;;
esac

case "${COGENT_CUBIN_COMPILE_OPT}" in
    DEFAULT|A100|H100|H200)
        ;;
    *)
        echo "[Error] Unsupported COGENT_CUBIN_COMPILE_OPT: ${COGENT_CUBIN_COMPILE_OPT}" >&2
        echo "[Error] Use one of: DEFAULT, A100, H100, H200" >&2
        exit 1
        ;;
esac

mkdir -p "${RESULT_DIR}"

echo "=========================================="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID:-local}"
echo "SLURM_NODELIST = ${SLURM_NODELIST:-local}"
echo "=========================================="
echo "[Slurm] Submit dir   : ${SLURM_SUBMIT_DIR:-none}"
echo "[Slurm] Script dir   : ${SCRIPT_DIR}"
echo "[Slurm] Repo root    : ${REPO_ROOT}"
echo "[Slurm] Python driver: ${PY_DRIVER}"
echo "[Slurm] Build dir    : ${CMAKE_BUILD_DIR}"
echo "[Slurm] Build target : ${CMAKE_TARGET}"
echo "[Slurm] GPU target   : ${GPU_TARGET:-env/default}"
echo "[Slurm] Cogent cubin : ${COGENT_CUBIN_COMPILE_OPT}"
echo "[Slurm] Binary       : ${BINARY_PATH}"
echo "[Slurm] Equation set : ${START_EQ}..${END_EQ}"
echo "[Slurm] Backend      : ${BACKEND}"
echo "[Slurm] Verify flag  : ${VERIFY_FLAG}"
echo "[Slurm] Result dir   : ${RESULT_DIR}"
echo "[Slurm] Host         : $(hostname)"
echo "[Slurm] Job ID       : ${SLURM_JOB_ID:-local}"
echo "[Slurm] Job GPUs     : ${SLURM_JOB_GPUS:-unset}"
echo "[Slurm] CUDA visible : ${CUDA_VISIBLE_DEVICES:-unset}"

if [[ -n "${SLURM_JOB_ID:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" && "${ALLOW_UNBOUND_GPU:-0}" != "1" ]]; then
    echo "[Warning] CUDA_VISIBLE_DEVICES is not set in the batch shell." >&2
    echo "[Warning] The benchmark will run through srun so the job step can apply GPU binding." >&2
fi

if [[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" && "${ALLOW_UNBOUND_GPU:-0}" != "1" ]]; then
    IFS=',' read -r -a visible_gpu_list <<< "${CUDA_VISIBLE_DEVICES}"
    if [[ "${#visible_gpu_list[@]}" -ne 1 ]]; then
        echo "[Warning] Expected one visible GPU in the batch shell, but CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
        echo "[Warning] The benchmark will run through srun so the job step can apply GPU binding." >&2
    fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[Slurm] nvidia-smi visible devices:"
    nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
fi

cd "${REPO_ROOT}"

rm -rf "${CMAKE_BUILD_DIR}"

echo "[Slurm] Configuring CMake project"
cmake -S "${REPO_ROOT}" -B "${CMAKE_BUILD_DIR}" \
    -DCOGENT_CUBIN_COMPILE_OPT="${COGENT_CUBIN_COMPILE_OPT}" \
    ${CMAKE_CONFIGURE_ARGS}

echo "[Slurm] Building ${CMAKE_TARGET}"
cmake --build "${CMAKE_BUILD_DIR}" --target "${CMAKE_TARGET}" ${CMAKE_BUILD_ARGS}

echo "[Slurm] Running benchmark driver"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 \
        bash -c 'echo "[Slurm step] CUDA visible : ${CUDA_VISIBLE_DEVICES:-unset}"; exec "$@"' bash \
        python3 -u "${PY_DRIVER}" \
        --binary "${BINARY_PATH}" \
        --output-dir "${RESULT_DIR}" \
        --start "${START_EQ}" \
        --end "${END_EQ}" \
        --backend "${BACKEND}" \
        "${PYTHON_ARGS[@]}"
else
    python3 -u "${PY_DRIVER}" \
        --binary "${BINARY_PATH}" \
        --output-dir "${RESULT_DIR}" \
        --start "${START_EQ}" \
        --end "${END_EQ}" \
        --backend "${BACKEND}" \
        "${PYTHON_ARGS[@]}"
fi

echo "[Slurm] Done. Results saved under ${RESULT_DIR}"
