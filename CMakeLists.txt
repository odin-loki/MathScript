cmake_minimum_required(VERSION 3.28)
project(MathScript VERSION 0.1.0 LANGUAGES CXX)

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
include(cmake/version.cmake)
include(cmake/coverage.cmake)

option(MS_BUILD_PLUGIN "Build Clang enforcement plugin (requires LLVM)" OFF)

if(MS_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

add_subdirectory(src)

if(MS_BUILD_TESTS)
    add_subdirectory(tests)
endif()

include(cmake/install.cmake)
