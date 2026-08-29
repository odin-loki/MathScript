# MathScript Compiler Flags
# Applied to all targets unconditionally

# Base warning flags
set(MS_CXX_FLAGS
    -Wall -Wextra -Wpedantic -Werror
    -Wconversion
    -Wshadow
    -Wnull-dereference
    -Wdouble-promotion
    -Wimplicit-fallthrough
)

# Debug flags
set(MS_CXX_FLAGS_DEBUG
    -O0 -g3
    -fno-omit-frame-pointer
)

# RelWithDebInfo flags
set(MS_CXX_FLAGS_RELWITHDEBINFO
    -O2 -g
    -fsanitize=undefined
)

# Release flags
set(MS_CXX_FLAGS_RELEASE
    -O3
    -flto=thin
    -fsanitize=undefined
    -fstack-protector-strong
    -D_FORTIFY_SOURCE=3
)

# Shipping flags
set(MS_CXX_FLAGS_SHIPPING
    -O3
    -flto=full
    -fprofile-use
    -fstack-protector-strong
    -D_FORTIFY_SOURCE=3
)

# Global flags - exceptions and RTTI off (GCC/Clang CXX only; do not pass to nvcc).
if(NOT MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
        $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
        $<$<COMPILE_LANGUAGE:CXX>:-fno-strict-aliasing>
    )
endif()

# Linker flags for Release/Shipping on Linux
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_link_options(
        -Wl,-z,relro,-z,now
    )
    # -pie is for executables only. Shared modules (ms_plugin) must not receive
    # it: Clang+lld warns "argument unused during compilation: '-pie'".
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie")
    find_program(MS_LLD_LINKER NAMES ld.lld lld)
    if(MS_LLD_LINKER)
        add_link_options(-fuse-ld=lld)
    else()
        message(STATUS "lld not found; using the default system linker")
    endif()
endif()

# Debug/coverage/ASan trees link hundreds of test executables. Concurrent links
# on GitHub-hosted runners exhaust disk (ENOSPC) and can bus-error ld.
if(NOT MSVC AND CMAKE_GENERATOR MATCHES "Ninja")
    if(MS_ENABLE_COVERAGE OR MS_ENABLE_ASAN OR CMAKE_BUILD_TYPE STREQUAL "Debug")
        cmake_host_system_information(RESULT _ms_cores QUERY NUMBER_OF_LOGICAL_CORES)
        if(NOT _ms_cores OR _ms_cores LESS 1)
            set(_ms_cores 4)
        endif()
        set(_ms_link_jobs 2)
        if(MS_ENABLE_COVERAGE)
            set(_ms_link_jobs 1)
        endif()
        set_property(GLOBAL PROPERTY JOB_POOLS
            compile_jobs=${_ms_cores} link_jobs=${_ms_link_jobs})
        set(CMAKE_JOB_POOL_COMPILE compile_jobs)
        set(CMAKE_JOB_POOL_LINK link_jobs)
    endif()
endif()

# Plugin — loaded for all non-plugin targets (master plan §3.4).
# Call ms_enable_plugin_enforcement() from root CMakeLists after add_subdirectory(src).
function(ms_enable_plugin_enforcement)
    if(NOT MS_BUILD_PLUGIN OR NOT TARGET ms_plugin)
        return()
    endif()
    get_property(_ms_targets GLOBAL PROPERTY TARGETS)
    foreach(_ms_t ${_ms_targets})
        if(_ms_t STREQUAL "ms_plugin" OR _ms_t STREQUAL "test_plugin_smoke")
            continue()
        endif()
        get_target_property(_ms_type ${_ms_t} TYPE)
        if(_ms_type STREQUAL "INTERFACE_LIBRARY" OR _ms_type STREQUAL "UTILITY")
            continue()
        endif()
        target_compile_options(${_ms_t} PRIVATE "-fplugin=$<TARGET_FILE:ms_plugin>")
    endforeach()
endfunction()