add_library(cogent STATIC
    src/cogent.cpp
)

set_target_properties(cogent PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

find_package(nlohmann_json CONFIG QUIET)

target_include_directories(cogent PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
)

if (nlohmann_json_FOUND)
    target_link_libraries(cogent PUBLIC nlohmann_json::nlohmann_json)
else()
    target_include_directories(cogent PUBLIC
        ${CMAKE_SOURCE_DIR}/external
    )
endif()

target_link_libraries(cogent PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::nvrtc
)
