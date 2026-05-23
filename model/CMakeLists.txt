add_executable(bench_rand_problem
    bench_rand_problem.cpp
)

target_link_libraries(bench_rand_problem PRIVATE tcont)

target_include_directories(bench_rand_problem PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/backends/cogent/include
    ${CMAKE_SOURCE_DIR}/backends/ttgt_cutt/include
    ${CMAKE_SOURCE_DIR}/external/cutt/src
    ${CMAKE_SOURCE_DIR}/external
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_rand_problem PRIVATE OpenMP::OpenMP_CXX)
endif()
