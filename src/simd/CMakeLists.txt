add_library(ms_simd STATIC isa.cpp vector_ops.cpp)
target_include_directories(ms_simd PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_simd PUBLIC ms_core PRIVATE xsimd)
target_compile_features(ms_simd PUBLIC cxx_std_23)

# Compile vector_ops.cpp as AVX2 so xsimd::batch<double> is 4-wide. MS_ENABLE_AVX512
# still enables Kernel::Avx512 reporting and the dedicated avx512_dgemm TU.
# Compiling these as AVX-512 would SIGILL on machines without the ISA (CI already
# passes -DMS_ENABLE_AVX512=OFF).
#
# isa.cpp is deliberately excluded and built at baseline. It is the file that
# decides whether AVX is safe to use on this host -- reading CPUID.1:ECX.OSXSAVE
# and then XCR0 -- and it is called unconditionally, before anything has been
# established. Compiling it with -mavx2 lets the compiler put an AVX instruction
# into the routine whose entire job is to determine whether AVX instructions will
# fault, which on a host where the OS has not enabled YMM state means detect_isa()
# raises SIGILL instead of returning "no AVX here". That is the same defect the
# XCR0 check exists to prevent, one level further down.
if(MSVC)
    set_source_files_properties(vector_ops.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(vector_ops.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()
