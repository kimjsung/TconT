#pragma once
#include <nvrtc.h>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include <vector>

enum class CubinCompileOptFlag {
    Default,
    A100,
    H100,
    H200
};

inline CubinCompileOptFlag compile_to_cubin_default_flag()
{
#if defined(COGENT_CUBIN_COMPILE_OPT_H100)
    return CubinCompileOptFlag::H100;
#elif defined(COGENT_CUBIN_COMPILE_OPT_H200)
    return CubinCompileOptFlag::H200;
#elif defined(COGENT_CUBIN_COMPILE_OPT_A100)
    return CubinCompileOptFlag::A100;
#else
    return CubinCompileOptFlag::Default;
#endif
}

inline std::string compile_to_cubin_arch(CubinCompileOptFlag flag)
{
    switch (flag) {
        case CubinCompileOptFlag::Default:
            return "sm_89";
        case CubinCompileOptFlag::A100:
            return "sm_80";
        case CubinCompileOptFlag::H100:
        case CubinCompileOptFlag::H200:
            return "sm_90";
    }

    return "sm_89";
}

inline std::vector<std::string> compile_to_cubin_opts(CubinCompileOptFlag flag)
{
    const std::string arch = "-arch=" + compile_to_cubin_arch(flag);

    switch (flag) {
        case CubinCompileOptFlag::Default:
            return {
                arch,
                "-I/usr/local/cuda/include",
                "--std=c++20",
                "--use_fast_math",
            };
        case CubinCompileOptFlag::A100:
            return {
                arch,
                "-I/apps/cuda/12.9.1/include",
                "--std=c++20",
                "--use_fast_math",
            };
        case CubinCompileOptFlag::H100:
        case CubinCompileOptFlag::H200:
            return {
                arch,
                "-I/apps/cuda/12.9.1/include",
                "--std=c++20",
                "--use_fast_math",
            };
    }

    return {};
}

inline void compile_to_cubin(
    const std::string& cu_path,
    const std::string& cubin_path,
    CubinCompileOptFlag flag = compile_to_cubin_default_flag())
{
    std::ifstream f(cu_path);
    if (!f) throw std::runtime_error("Cannot open: " + cu_path);
    
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string src = ss.str();

    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, src.c_str(), cu_path.c_str(), 0, nullptr, nullptr);

    const std::vector<std::string> compile_opts = compile_to_cubin_opts(flag);
    std::vector<const char*> raw_opts;
    raw_opts.reserve(compile_opts.size());
    for (const std::string& opt : compile_opts) {
        raw_opts.push_back(opt.c_str());
    }

    nvrtcResult res = nvrtcCompileProgram(
        prog,
        static_cast<int>(raw_opts.size()),
        raw_opts.data());

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
