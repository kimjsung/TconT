cmake_minimum_required(VERSION 3.18)
project(TconT_Workspace LANGUAGES CXX CUDA)

# Add your library first so benchmarks can find it
add_subdirectory(tcont)
add_subdirectory(benchmarks)

