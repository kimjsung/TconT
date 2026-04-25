#pragma once
#include <nvrtc.h>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <filesystem>

inline void compile_to_cubin(const std::string& cu_path, const std::string& cubin_path)
{
    std::ifstream f(cu_path);
    if (!f) throw std::runtime_error("Cannot open: " + cu_path);
    
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string src = ss.str();

    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, src.c_str(), cu_path.c_str(), 0, nullptr, nullptr);

    const char* opts[] = {
        "-arch=sm_89",
        "-I/usr/local/cuda/include",
        "--std=c++20",
        "--use_fast_math"
    };
    
    // For A100
    /*const char* opts[] = {
        "-arch=sm_80",
        "-I/apps/cuda/12.9.1/include",
        "--std=c++20",
        "--use_fast_math"
    };*/

    nvrtcResult res = nvrtcCompileProgram(prog, 4, opts);

    if (res != NVRTC_SUCCESS) {
        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        std::string log(logSize, '\0');
        nvrtcGetProgramLog(prog, log.data());
        nvrtcDestroyProgram(&prog);
        throw std::runtime_error("nvrtc compile error:\n" + log);
    }

    size_t cubinSize;
    nvrtcGetCUBINSize(prog, &cubinSize);

    if (cubinSize == 0)
        throw std::runtime_error("cubinSize is 0 — creation of CUBIN Fail");
    
    std::string cubin(cubinSize, '\0');
    nvrtcGetCUBIN(prog, cubin.data());
    nvrtcDestroyProgram(&prog);

    std::ofstream out(cubin_path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write: " + cubin_path);
    out.write(cubin.data(), cubinSize);
}