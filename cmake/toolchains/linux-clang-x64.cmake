# Linux x86_64 Clang Toolchain for MathScript
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER clang)
set(CMAKE_LINKER ld.lld)

# Use LLVM ar
if(DEFINED LLVM_AR)
    set(CMAKE_AR "${LLVM_AR}")
endif()

# Set default architecture
if(NOT CMAKE_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=x86-64")
endif()