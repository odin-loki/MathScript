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

# Global flags - exceptions and RTTI off
add_compile_options(
    -fno-exceptions
    -fno-rtti
    -fno-strict-aliasing
)

# Linker flags for Release/Shipping on Linux
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_link_options(
        -Wl,-z,relro,-z,now
        -pie
        -fuse-ld=lld
    )
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