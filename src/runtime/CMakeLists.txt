# MathScript Runtime Library

set(MS_RUNTIME_CPU_SOURCES
    ${CMAKE_SOURCE_DIR}/src/cpu/blas.cpp
    cpu/blas_dgemm.cpp
    cpu/blas_sgemm.cpp
    cpu/blas_dger.cpp
    cpu/blas_dsyrk_dtrsm.cpp
    cpu/lapack_dpotrf.cpp
    cpu/lapack_dgetrf.cpp
    cpu/lapack_dgesv.cpp
    cpu/lapack_dgeqrf.cpp
    cpu/lapack_dgels.cpp
    cpu/lapack_dsyev.cpp
    cpu/lapack_dgebrd.cpp
    cpu/lapack_dorgbr.cpp
    cpu/lapack_dbdsqr.cpp
    cpu/lapack_dgesvd.cpp
)

if(MS_ENABLE_AVX512)
    list(APPEND MS_RUNTIME_CPU_SOURCES cpu/avx512_dgemm.cpp)
else()
    list(APPEND MS_RUNTIME_CPU_SOURCES cpu/avx512_dgemm_stub.cpp)
endif()

if(MS_ENABLE_AVX2)
    list(APPEND MS_RUNTIME_CPU_SOURCES cpu/avx2_dgemm.cpp cpu/avx2_sgemm.cpp)
else()
    list(APPEND MS_RUNTIME_CPU_SOURCES cpu/avx2_dgemm_stub.cpp cpu/avx2_sgemm_stub.cpp)
endif()

add_library(ms_runtime STATIC
    topology.cpp
    thread_pool.cpp
    dispatch.cpp
    load_balancer.cpp
    repro.cpp
    ${MS_RUNTIME_CPU_SOURCES}
)

target_include_directories(ms_runtime PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(ms_runtime PUBLIC ms_core ms_cuda ms_frameworks ms_simd)
target_compile_features(ms_runtime PUBLIC cxx_std_23)

# Only the kernel translation units get the wide ISA flags. Everything that runs
# before the runtime ISA check -- the dispatcher, and ms::simd::detect_isa itself --
# stays at baseline, because a detection routine compiled with the instructions it
# is meant to be deciding about can fault before it returns an answer.
if(MS_ENABLE_AVX512)
    if(MSVC)
        set_source_files_properties(cpu/avx512_dgemm.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    else()
        set_source_files_properties(cpu/avx512_dgemm.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512dq")
    endif()
endif()

if(MS_ENABLE_AVX2)
    if(MSVC)
        set_source_files_properties(cpu/avx2_dgemm.cpp cpu/avx2_sgemm.cpp
                                    PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    else()
        set_source_files_properties(cpu/avx2_dgemm.cpp cpu/avx2_sgemm.cpp
                                    PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    endif()
endif()
