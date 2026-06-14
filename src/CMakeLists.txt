# src/CMakeLists.txt

# Collect source files
file(GLOB TCONT_SOURCES "*.cpp")

# Define the library
add_library(tcont SHARED ${TCONT_SOURCES})

find_package(nlohmann_json CONFIG QUIET)

# Include directories
target_include_directories(tcont PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if (nlohmann_json_FOUND)
    target_link_libraries(tcont PRIVATE nlohmann_json::nlohmann_json)
else()
    target_include_directories(tcont PRIVATE
        ${CMAKE_SOURCE_DIR}/external
    )
endif()

target_compile_definitions(tcont PRIVATE
    TCONT_DEFAULT_MODEL_JSON="${CMAKE_SOURCE_DIR}/final_model/lgbm_small_i120_depth3_leaf15_lr005_l2_1_export.json"
)

# Link the backends (assuming they define targets 'cogent' and 'ttgt_cutt')
target_link_libraries(tcont PRIVATE cogent ttgt_cutt)


#
target_link_libraries(tcont PUBLIC CUDA::cudart)


# Export the library for the benchmarks
set_target_properties(tcont PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
