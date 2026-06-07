# Linux x86_64 Clang + CUDA Toolchain for MathScript
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER clang)
set(CMAKE_LINKER ld.lld)

# CUDA host compiler
set(CMAKE_CUDA_HOST_COMPILER clang++)

# CUDA compute capabilities (Ampere, Ada, Hopper)
set(CMAKE_CUDA_ARCHITECTURES "70;80;86;89;90")

# NVCC wrapper settings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fno-exceptions -Xcompiler=-fno-rtti")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# NCCL for multi-GPU
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-D_NCCL_")