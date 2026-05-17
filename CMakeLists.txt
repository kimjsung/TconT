cmake_minimum_required(VERSION 3.18)
project(TconT_Workspace LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80 89)

set(COGENT_CUBIN_COMPILE_OPT "DEFAULT" CACHE STRING "Cogent NVRTC cubin compile preset (DEFAULT or A100)")
set_property(CACHE COGENT_CUBIN_COMPILE_OPT PROPERTY STRINGS DEFAULT A100)


find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)


# Add bundled third-party dependencies and backends
add_subdirectory(external/cutt)
add_subdirectory(backends/cogent)
add_subdirectory(backends/ttgt_cutt)

# Add the main library
add_subdirectory(src)

# Add benchmarks 
add_subdirectory(benchmarks)
