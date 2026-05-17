
set(CUTT_SOURCE_FILES
    calls.h
    CudaMem.cpp
    CudaMem.h
    CudaMemcpy.cu
    CudaMemcpy.h
    CudaUtils.cu
    CudaUtils.h
    cutt.cpp
    cutt.h
    cuttGpuModel.cpp
    cuttGpuModel.h
    cuttGpuModelKernel.cu
    cuttGpuModelKernel.h
    cuttkernel.cu
    cuttkernel.h
    cuttplan.h
    cuttplan.cpp
    cuttTimer.cpp
    cuttTimer.h
    cuttTypes.h
    int_vector.h
    TensorTester.cu
    TensorTester.h
    LRUCache.h)


add_library(cutt ${CUTT_SOURCE_FILES})
target_include_directories(cutt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(cutt PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(ENABLE_UMPIRE)
    target_link_libraries(cutt umpire)
    target_compile_definitions(cutt PUBLIC CUTT_HAS_UMPIRE -DCUTT_USES_THIS_UMPIRE_ALLOCATOR=${CUTT_USES_THIS_UMPIRE_ALLOCATOR})
endif()

# install library and include
install(TARGETS cutt
        ARCHIVE
            DESTINATION lib
        LIBRARY
            DESTINATION lib
        )
install(FILES cutt.h
        DESTINATION include
        )

if(CUTT_BUILD_TOOLS)
    add_executable(cutt_bench cutt_bench.cpp)
    target_link_libraries(cutt_bench PUBLIC cutt)

    add_executable(cutt_test cutt_test.cpp)
    target_link_libraries(cutt_test PUBLIC cutt)
endif()
