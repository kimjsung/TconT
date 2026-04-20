#pragma once
#include <nvrtc.h>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <filesystem>

inline void compile_to_cubin(const std::string& cu_path, const std::string& cubin_path)
{
    // 1) .cu 소스 읽기
    std::ifstream f(cu_path);
    if (!f) throw std::runtime_error("Cannot open: " + cu_path);
    
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string src = ss.str();

    // 2) nvrtc 프로그램 생성 + 컴파일
    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, src.c_str(), cu_path.c_str(), 0, nullptr, nullptr);

    const char* opts[] = {
        "-arch=sm_89",                        // gpu architecture
        "-I/usr/local/cuda/include",               // CUDA 헤더
        // "-I/usr/include",                          // 시스템 헤더 (nvcc에서 -I/usr/include 쓰고 있으므로)
        "--std=c++20"
    };
    
    // For A100
    /*const char* opts[] = {
        "-arch=sm_80",                                                           // gpu architecture
        "-I/apps/cuda/12.9.1/include",
        "--std=c++20",
        "--use_fast_math"
    };*/

    nvrtcResult res = nvrtcCompileProgram(prog, 3, opts);

    if (res != NVRTC_SUCCESS) {
        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        std::string log(logSize, '\0');
        nvrtcGetProgramLog(prog, log.data());
        nvrtcDestroyProgram(&prog);
        throw std::runtime_error("nvrtc compile error:\n" + log);
    }

    // 3) cubin 추출 + 저장
    size_t cubinSize;
    nvrtcGetCUBINSize(prog, &cubinSize);

    if (cubinSize == 0)
        throw std::runtime_error("cubinSize is 0 — CUBIN 생성 실패");
    
    std::string cubin(cubinSize, '\0');
    nvrtcGetCUBIN(prog, cubin.data());
    nvrtcDestroyProgram(&prog);

    std::ofstream out(cubin_path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write: " + cubin_path);
    out.write(cubin.data(), cubinSize);
}