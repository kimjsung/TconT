# Define a benchmark executable
add_executable(bench_tccg bench_tccg.cpp)

# Link against our main library
target_link_libraries(bench_tccg PRIVATE tcont)

# Optional: Add OpenMP or other performance profiling tools here
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_tccg PRIVATE OpenMP::OpenMP_CXX)
endif()


