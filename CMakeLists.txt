cmake_minimum_required(VERSION 3.28)
project(MathScript VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive- /Zc:__cplusplus)
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
