add_library(ttgt_cutt STATIC
    src/ttgt_cutt_backend.cpp
    src/ttgt_cutt.cpp
    src/ttgt_cutt_models.cpp
    src/ttgt/ttgt_ttlg.cpp
    src/ttgt/algs/enum_helper.cpp
    src/ttgt/algs/enumeration.cpp
    src/ttgt/algs/tiling_scratch.cpp
    src/ttgt/algs/ttgt_lottgt.cpp
    src/ttgt/lottgt/lottgt_tiles.cpp
    src/ttgt/lottgt/ttgt_ttlg_constraints.cpp
    src/ttgt/models/model_base.cpp
    src/ttgt/models/model_copy.cpp
    src/ttgt/models/model_gemm.cpp
    src/ttgt/models/model_ttlg.cpp
)

set_target_properties(ttgt_cutt PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(ttgt_cutt PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(ttgt_cutt PUBLIC
    cutt
    CUDA::cublas
    CUDA::cudart
    OpenMP::OpenMP_CXX
)
