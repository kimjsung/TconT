add_executable(bench_tccg
    bench_tccg.cpp
    tccg_cases.cpp
    tccg_verify.cpp
)

target_link_libraries(bench_tccg PRIVATE tcont cogent ttgt_cutt CUDA::cublas)

target_include_directories(bench_tccg PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/backends/cogent/include
    ${CMAKE_SOURCE_DIR}/backends/ttgt_cutt/include
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_tccg PRIVATE OpenMP::OpenMP_CXX)
endif()
