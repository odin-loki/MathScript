# Windows Clang-cl Toolchain for MathScript
set(CMAKE_CXX_COMPILER clang-cl)
set(CMAKE_C_COMPILER clang-cl)
set(CMAKE_LINKER lld-link)

# MSVC ABI for CUDA interop
set(CMAKE_CXX_FLAGS_INIT "/EHs- /GR- /W4")

# mimalloc Windows override
add_compile_definitions(MI_OVERRIDE=1)

# Use LLVM archiver
if(DEFINED LLVM_AR)
    set(CMAKE_AR "${LLVM_AR}")
endif()