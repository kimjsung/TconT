#!/bin/bash
#SBATCH -J TCCG_ALL
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH -o all.out
#SBATCH -e %x.%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --comment etc

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "${SCRIPT_DIR}/logs"

# Uncomment and adjust these lines if your cluster uses environment modules.
module purge
module load gcc/11.5.0 nvidia_hpc_sdk/24.5 python/3.14.2

CMAKE_BUILD_DIR="${CMAKE_BUILD_DIR:-${REPO_ROOT}/build}"
CMAKE_TARGET="${CMAKE_TARGET:-bench_tccg}"
CMAKE_CONFIGURE_ARGS="${CMAKE_CONFIGURE_ARGS:-}"
CMAKE_BUILD_ARGS="${CMAKE_BUILD_ARGS:--j8}"
BINARY_PATH="${BINARY_PATH:-${CMAKE_BUILD_DIR}/benchmarks/tccg/bench_tccg}"
START_EQ="${START_EQ:-0}"
END_EQ="${END_EQ:-48}"
VERIFY_FLAG="${VERIFY_FLAG:---verify}"
RESULT_ROOT="${RESULT_ROOT:-${SCRIPT_DIR}/results}"
RESULT_DIR="${RESULT_DIR:-${RESULT_ROOT}/job_${SLURM_JOB_ID}}"

mkdir -p "${RESULT_DIR}"

echo "[Slurm] Repo root    : ${REPO_ROOT}"
echo "[Slurm] Build dir    : ${CMAKE_BUILD_DIR}"
echo "[Slurm] Build target : ${CMAKE_TARGET}"
echo "[Slurm] Binary       : ${BINARY_PATH}"
echo "[Slurm] Equation set : ${START_EQ}..${END_EQ}"
echo "[Slurm] Verify flag  : ${VERIFY_FLAG}"
echo "[Slurm] Result dir   : ${RESULT_DIR}"
echo "[Slurm] Host         : $(hostname)"
echo "[Slurm] Job ID       : ${SLURM_JOB_ID:-local}"

cd "${REPO_ROOT}"

if [[ ! -f "${CMAKE_BUILD_DIR}/CMakeCache.txt" ]]; then
    echo "[Slurm] Configuring CMake project"
    cmake -S "${REPO_ROOT}" -B "${CMAKE_BUILD_DIR}" ${CMAKE_CONFIGURE_ARGS}
fi

echo "[Slurm] Building ${CMAKE_TARGET}"
cmake --build "${CMAKE_BUILD_DIR}" --target "${CMAKE_TARGET}" ${CMAKE_BUILD_ARGS}

python3 "${SCRIPT_DIR}/run_tccg_benchmarks.py" \
    --binary "${BINARY_PATH}" \
    --output-dir "${RESULT_DIR}" \
    --start "${START_EQ}" \
    --end "${END_EQ}" \
    ${VERIFY_FLAG:+${VERIFY_FLAG}}

echo "[Slurm] Done. Results saved under ${RESULT_DIR}"
