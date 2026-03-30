# src/CMakeLists.txt

# Collect source files
file(GLOB TCONT_SOURCES "*.cpp")

# Define the library
add_library(tcont SHARED ${TCONT_SOURCES})

# Include directories
target_include_directories(tcont PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link the backends (assuming they define targets 'cogent' and 'ttgt_cutt')
target_link_libraries(tcont PRIVATE cogent ttgt_cutt)


#
target_link_libraries(tcont PUBLIC CUDA::cudart)


# Export the library for the benchmarks
set_target_properties(tcont PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)


