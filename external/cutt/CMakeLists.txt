include(CheckFunctionExists)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

option(CUTT_ENABLE_NVTOOLS "Enable nvvp profiling of CPU code" OFF)
option(CUTT_ENABLE_NO_ALIGNED_ALLOC "Enable aligned_alloc() function implemented in cuTT" OFF)
option(CUTT_ENABLE_UMPIRE "Enable umpire for memory management" OFF)
option(CUTT_BUILD_TOOLS "Build vendored cuTT benchmark/test executables" OFF)

if(CUTT_ENABLE_NVTOOLS)
    add_definitions(-DENABLE_NVTOOLS)
    list(APPEND CUDA_NVCC_FLAGS -lnvToolsExt)
    link_libraries(-lnvToolsExt)
endif()

if(CUTT_ENABLE_NO_ALIGNED_ALLOC)
    add_definitions(-DNO_ALIGNED_ALLOC)
else()
    CHECK_FUNCTION_EXISTS(aligned_alloc HAVE_ALIGNED_ALLOC)
    if(NOT HAVE_ALIGNED_ALLOC)
        add_definitions(-DNO_ALIGNED_ALLOC)
    endif()
endif()

if(CUTT_ENABLE_UMPIRE)
    find_package(umpire REQUIRED)
    if (NOT CUTT_USES_THIS_UMPIRE_ALLOCATOR)
        set(CUTT_USES_THIS_UMPIRE_ALLOCATOR "UM")
    endif()
endif()

add_subdirectory(src)
