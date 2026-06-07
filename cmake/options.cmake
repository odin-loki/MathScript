# MathScript Build Options

# GUI build
option(MS_BUILD_GUI "Build Qt6 IDE" OFF)

# Tests
option(MS_BUILD_TESTS "Build test suite" ON)

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
