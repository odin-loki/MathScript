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

# Remove plugin flag (plugin build disabled)
# Plugin path for non-plugin targets (commented out)
# if(NOT MS_BUILDING_PLUGIN)
#     add_compile_options(-fplugin=${MS_PLUGIN_PATH})
# endif()