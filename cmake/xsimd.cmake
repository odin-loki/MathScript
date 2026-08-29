# Header-only xsimd (xtensor-stack) for portable SIMD intrinsics.
# Prefer a vendored tree when present; otherwise FetchContent a pinned tag.

set(_MS_XSIMD_TAG "13.2.0")
set(_MS_VENDOR_XSIMD "${CMAKE_SOURCE_DIR}/vendor/xsimd/include/xsimd/xsimd.hpp")

if(EXISTS "${_MS_VENDOR_XSIMD}")
    if(NOT TARGET xsimd)
        add_library(xsimd INTERFACE)
        add_library(xsimd::xsimd ALIAS xsimd)
        target_include_directories(xsimd SYSTEM INTERFACE
            "${CMAKE_SOURCE_DIR}/vendor/xsimd/include")
    endif()
    message(STATUS "MathScript xsimd: vendored vendor/xsimd")
else()
    include(FetchContent)
    set(BUILD_TESTS OFF CACHE BOOL "Disable xsimd tests" FORCE)
    FetchContent_Declare(
        xsimd
        URL "https://github.com/xtensor-stack/xsimd/archive/refs/tags/${_MS_XSIMD_TAG}.tar.gz"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(xsimd)
    message(STATUS "MathScript xsimd ${_MS_XSIMD_TAG} via FetchContent")
endif()

if(TARGET xsimd)
    get_target_property(_ms_xsimd_inc xsimd INTERFACE_INCLUDE_DIRECTORIES)
    if(_ms_xsimd_inc)
        set_property(TARGET xsimd PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_ms_xsimd_inc}")
    endif()
endif()
