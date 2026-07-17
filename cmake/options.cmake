# MathScript Build Options

# GUI build
option(MS_BUILD_GUI "Build Qt6 IDE" OFF)

# Tests
option(MS_BUILD_TESTS "Build test suite" ON)

if(EXISTS "${CMAKE_SOURCE_DIR}/vendor/googletest/CMakeLists.txt")
    set(_MS_VENDOR_GTEST_DEFAULT ON)
else()
    set(_MS_VENDOR_GTEST_DEFAULT OFF)
endif()
option(MS_USE_VENDOR_GTEST
    "Use vendored GoogleTest from vendor/googletest instead of FetchContent"
    ${_MS_VENDOR_GTEST_DEFAULT})

# Performance benchmarks (Google Benchmark)
option(MS_BUILD_BENCHMARKS "Build performance benchmarks" OFF)

# CUDA
option(MS_ENABLE_CUDA "Enable CUDA GPU backend" OFF)

# MPI
option(MS_ENABLE_MPI "Enable distributed MPI" OFF)

# NCCL for multi-GPU
option(MS_ENABLE_NCCL "Enable multi-GPU NCCL" ON)

# AVX-512
option(MS_ENABLE_AVX512 "Enable AVX-512 kernels" ON)

# Force ASan
option(MS_ENABLE_ASAN "Force ASan regardless of build type" OFF)

# Coverage
option(MS_ENABLE_COVERAGE "Enable coverage instrumentation" OFF)

# libFuzzer targets (requires Clang or MSVC with /fsanitize=fuzzer)
option(MS_BUILD_FUZZ "Build libFuzzer targets" OFF)

# Clang AST enforcement plugin (requires LLVM/Clang dev packages)
option(MS_BUILD_PLUGIN "Build Clang enforcement plugin (requires LLVM)" OFF)

# LLVM ORC JIT backend (requires LLVM dev packages; Linux CI optional job)
option(MS_BUILD_JIT "Build LLVM ORC JIT backend when LLVM is available" OFF)

# Use libc++ on Linux CI/dev when libstdc++ lacks full C++23 (e.g. std::expected)
option(MS_USE_LIBCXX "Use libc++ standard library on Linux" OFF)

# Verbose dispatch logging
option(MS_VERBOSE_DISPATCH "Log dispatch decisions at runtime" OFF)

# CUDA architecture targets
set(MS_CUDA_ARCHITECTURES "70;80;86;89;90"
    CACHE STRING "CUDA SM architectures to compile for")
set_property(CACHE MS_CUDA_ARCHITECTURES PROPERTY STRINGS
    "70" "80" "86" "89" "90" "70;80;86;89;90")

if(MS_ENABLE_AVX512)
    add_compile_definitions(MS_ENABLE_AVX512=1)
else()
    add_compile_definitions(MS_ENABLE_AVX512=0)
endif()

if(MS_ENABLE_ASAN)
    if(MS_COMPILER STREQUAL "clang" OR MS_COMPILER STREQUAL "gcc")
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
        add_link_options(-fsanitize=address,undefined)
        message(STATUS "AddressSanitizer + UBSan enabled")
    endif()
endif()

if(MS_BUILD_FUZZ AND NOT MSVC)
    add_compile_options(-fsanitize=fuzzer-no-link,address -g -O1 -fno-omit-frame-pointer)
    add_link_options(-fsanitize=fuzzer-no-link,address)
endif()

if(MS_USE_LIBCXX AND MS_PLATFORM STREQUAL "linux")
    add_compile_options(-stdlib=libc++)
    add_link_options(-stdlib=libc++ -lc++abi)
elseif(MS_PLATFORM STREQUAL "linux" AND MS_COMPILER STREQUAL "clang")
    add_compile_options(--gcc-toolchain=/usr -stdlib=libstdc++)
    add_compile_definitions(__cpp_concepts=202002L)
endif()
