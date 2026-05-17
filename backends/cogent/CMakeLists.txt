add_library(cogent STATIC
    src/cogent.cpp
)

set_target_properties(cogent PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

if (COGENT_CUBIN_COMPILE_OPT STREQUAL "DEFAULT")
    target_compile_definitions(cogent PUBLIC COGENT_CUBIN_COMPILE_OPT_DEFAULT)
elseif (COGENT_CUBIN_COMPILE_OPT STREQUAL "A100")
    target_compile_definitions(cogent PUBLIC COGENT_CUBIN_COMPILE_OPT_A100)
else()
    message(FATAL_ERROR "Unsupported COGENT_CUBIN_COMPILE_OPT='${COGENT_CUBIN_COMPILE_OPT}'. Use DEFAULT or A100.")
endif()

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
