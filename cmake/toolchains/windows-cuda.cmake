# Windows CUDA Toolchain for MathScript
set(CMAKE_CUDA_HOST_COMPILER cl)

# MSVC CUDA flags
set(CMAKE_CUDA_FLAGS
    "--compiler-options /EHs- /GR-"
    "-Xcompiler /W4"
)

# CUDA compute capabilities
set(CMAKE_CUDA_ARCHITECTURES "70;80;86;89;90")

# Use MSVC archiver
if(DEFINED MSVC_AR)
    set(CMAKE_AR "${MSVC_AR}")
endif()