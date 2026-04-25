add_library(cogent STATIC
    src/cogent.cpp
)

set_target_properties(cogent PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(cogent PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(cogent PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::nvrtc
)
