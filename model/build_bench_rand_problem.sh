#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ROOT_BUILD_DIR="${REPO_ROOT}/build_model_root"
MODEL_BUILD_DIR="${REPO_ROOT}/build_model_tools"
COGENT_CUBIN_COMPILE_OPT="A100"
JOBS="$(nproc)"
CUDA_ROOT="${CUDA_ROOT:-${CUDA_HOME:-/usr/local/cuda}}"

if [[ ! -d "${CUDA_ROOT}" ]] && command -v nvcc >/dev/null 2>&1; then
    CUDA_ROOT="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root-build-dir)
            ROOT_BUILD_DIR="$2"
            shift 2
            ;;
        --model-build-dir)
            MODEL_BUILD_DIR="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --cuda-root)
            CUDA_ROOT="$2"
            shift 2
            ;;
        --cogent-cubin-compile-opt)
            COGENT_CUBIN_COMPILE_OPT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ! -d "${CUDA_ROOT}/include" || ! -d "${CUDA_ROOT}/lib64" ]]; then
    echo "CUDA_ROOT does not look valid: ${CUDA_ROOT}" >&2
    echo "Set CUDA_ROOT or CUDA_HOME before running this script." >&2
    exit 1
fi

mkdir -p "${ROOT_BUILD_DIR}" "${MODEL_BUILD_DIR}"

cmake -S "${REPO_ROOT}" -B "${ROOT_BUILD_DIR}" \
    -DCOGENT_CUBIN_COMPILE_OPT="${COGENT_CUBIN_COMPILE_OPT}"
cmake --build "${ROOT_BUILD_DIR}" --target tcont -j "${JOBS}"

g++ -std=c++20 -O2 -fopenmp \
    "${REPO_ROOT}/model/bench_rand_problem.cpp" \
    -o "${MODEL_BUILD_DIR}/bench_rand_problem" \
    -I"${REPO_ROOT}/include" \
    -I"${REPO_ROOT}/backends/cogent/include" \
    -I"${REPO_ROOT}/backends/ttgt_cutt/include" \
    -I"${REPO_ROOT}/external" \
    -I"${REPO_ROOT}/external/cutt/src" \
    -I"${CUDA_ROOT}/include" \
    -L"${ROOT_BUILD_DIR}/src" \
    -ltcont \
    -L"${CUDA_ROOT}/lib64" \
    -lcudart \
    -Wl,-rpath,"${ROOT_BUILD_DIR}/src" \
    -Wl,-rpath,"${CUDA_ROOT}/lib64"
