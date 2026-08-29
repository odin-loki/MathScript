set(_MS_CUDA_SOURCES
    buffer.cpp
    blas.cpp
    fft.cpp
    solver.cpp
    elementwise.cpp
    sparse.cpp
    nvml.cpp
    nccl.cpp
)

# kernels.cu is added after CUDAToolkit is found (needs cuda_runtime).

add_library(ms_cuda STATIC ${_MS_CUDA_SOURCES})
target_include_directories(ms_cuda PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_cuda PUBLIC ms_core PRIVATE ms_simd)
target_compile_features(ms_cuda PUBLIC cxx_std_23)

set(_MS_CUDA_TOOLKIT_OK OFF)
if(MS_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(_MS_CUDA_TOOLKIT_OK ON)
        target_compile_definitions(ms_cuda PUBLIC MS_HAS_CUDA=1)
        target_link_libraries(ms_cuda PUBLIC CUDA::cudart CUDA::cublas CUDA::cufft CUDA::cusolver CUDA::cusparse)
        message(STATUS "MathScript CUDA toolkit found: ${CUDAToolkit_VERSION}")

        if(MS_ENABLE_NCCL)
            find_path(NCCL_INCLUDE_DIR NAMES nccl.h)
            find_library(NCCL_LIBRARY NAMES nccl)
            if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
                target_compile_definitions(ms_cuda PUBLIC MS_HAS_NCCL=1)
                target_include_directories(ms_cuda PRIVATE ${NCCL_INCLUDE_DIR})
                target_link_libraries(ms_cuda PUBLIC ${NCCL_LIBRARY})
                message(STATUS "MathScript NCCL found: ${NCCL_LIBRARY}")
            else()
                target_compile_definitions(ms_cuda PUBLIC MS_HAS_NCCL=0)
                message(STATUS "MathScript NCCL not found; using stub (MS_HAS_NCCL=0)")
            endif()
        else()
            target_compile_definitions(ms_cuda PUBLIC MS_HAS_NCCL=0)
        endif()
    endif()
endif()

if(NOT _MS_CUDA_TOOLKIT_OK)
    target_compile_definitions(ms_cuda PUBLIC MS_HAS_CUDA=0 MS_HAS_NCCL=0)
endif()

if(MS_CUDA_COMPILER_FOUND AND _MS_CUDA_TOOLKIT_OK)
    target_sources(ms_cuda PRIVATE kernels.cu)
    set_target_properties(ms_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${MS_CUDA_ARCHITECTURES}"
    )
    set_source_files_properties(kernels.cu PROPERTIES LANGUAGE CUDA)
endif()
