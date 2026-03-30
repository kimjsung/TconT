add_library(cogent INTERFACE) # Use INTERFACE for header-only stubs
target_include_directories(cogent INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
