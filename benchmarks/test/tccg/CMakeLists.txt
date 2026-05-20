add_executable(bench_tccg_iter
    bench_tccg_iter.cpp
    ../../tccg/tccg_cases.cpp
    ../../tccg/tccg_verify.cpp
)

target_link_libraries(bench_tccg_iter PRIVATE tcont ttgt_cutt cogent CUDA::cublas)

target_include_directories(bench_tccg_iter PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../tccg
    ${CMAKE_SOURCE_DIR}/backends/cogent/include
    ${CMAKE_SOURCE_DIR}/backends/ttgt_cutt/include
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_tccg_iter PRIVATE OpenMP::OpenMP_CXX)
endif()
