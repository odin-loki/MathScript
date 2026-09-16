cmake_minimum_required(VERSION 3.28)
project(MathScript VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    # /utf-8 sets both the source and the execution character set to UTF-8. Without it
    # MSVC reads the source in the system codepage, and src/sym2/notation_unicode.cpp
    # is the first file in this tree whose non-ASCII bytes are inside string literals
    # rather than comments -- its `pi`, `sqrt` and `dot` glyphs would be decoded as
    # CP-1252 and reach the user as mojibake. That failure is invisible to the tests
    # that cover the printers: mangled bytes still balance their delimiters and are
    # still deterministic, so the suite would pass while Windows produced the wrong
    # characters. It also silences C4819 on the eight existing files with non-ASCII
    # comments, which was a warning about exactly this.
    add_compile_options(/W4 /permissive- /Zc:__cplusplus /utf-8)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.30)
        add_compile_options(/std:c++latest)
    endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(cmake/platform.cmake)
include(cmake/options.cmake)
include(cmake/compiler_flags.cmake)
include(cmake/verify_vendor.cmake)
include(cmake/version.cmake)
include(cmake/coverage.cmake)
include(cmake/jit.cmake)
include(cmake/xsimd.cmake)

set(MS_CUDA_COMPILER_FOUND OFF)
if(MS_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_ARCHITECTURES ${MS_CUDA_ARCHITECTURES})
        set(MS_CUDA_COMPILER_FOUND ON)
        message(STATUS "MathScript CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    else()
        message(STATUS "MathScript CUDA compiler not found; GPU kernels disabled")
    endif()
endif()

if(MS_BUILD_TESTS)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    if(MS_USE_VENDOR_GTEST AND EXISTS "${CMAKE_SOURCE_DIR}/vendor/googletest/CMakeLists.txt")
        add_subdirectory(vendor/googletest)
    else()
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(googletest)
    endif()
    enable_testing()
endif()

if(MS_BUILD_BENCHMARKS)
    include(FetchContent)
    FetchContent_Declare(
        benchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.8.5.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
endif()

# Is the TARGET x86? Not "is the compiler GCC-like" -- that is the mistake
# this replaces, and it is what sent -mavx2 -mfma to every aarch64 build
# GCC and Clang support, Apple Silicon and ARM Linux included, not just
# WebAssembly. Emscripten is tested first because its toolchain file has
# reported an x86 CMAKE_SYSTEM_PROCESSOR.
if(EMSCRIPTEN)
    set(MS_TARGET_X86 FALSE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86|x86)$")
    set(MS_TARGET_X86 TRUE)
else()
    set(MS_TARGET_X86 FALSE)
endif()
message(STATUS "MathScript: target is x86 = ${MS_TARGET_X86} (processor ${CMAKE_SYSTEM_PROCESSOR})")

add_subdirectory(src)

if(MS_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(MS_BUILD_PLUGIN)
    ms_enable_plugin_enforcement()
endif()

if(MS_BUILD_BENCHMARKS)
    add_subdirectory(tests/performance)
endif()

include(cmake/install.cmake)
