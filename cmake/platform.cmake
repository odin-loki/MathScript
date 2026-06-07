# Platform Detection for MathScript

# Detect platform
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(MS_PLATFORM "linux")
    set(MS_IS_WINDOWS 0)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(MS_PLATFORM "windows")
    set(MS_IS_WINDOWS 1)
else()
    set(MS_PLATFORM "unknown")
endif()

# Detect compiler
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(MS_COMPILER "clang")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(MS_COMPILER "msvc")
else()
    set(MS_COMPILER "gcc")
endif()

# Detect CUDA availability
if(DEFINED CUDAToolkit_FOUND)
    set(MS_HAS_CUDA ON)
else()
    set(MS_HAS_CUDA OFF)
endif()

# Detect MPI availability
if(DEFINED MPI_FOUND)
    set(MS_HAS_MPI ON)
else()
    set(MS_HAS_MPI OFF)
endif()

# Detect TBB availability
if(DEFINED TBB_FOUND)
    set(MS_HAS_TBB ON)
else()
    set(MS_HAS_TBB OFF)
endif()

# Output platform info
message(STATUS "MathScript Platform: ${MS_PLATFORM}")
message(STATUS "MathScript Compiler: ${MS_COMPILER}")
message(STATUS "MathScript CUDA: ${MS_HAS_CUDA}")
message(STATUS "MathScript MPI: ${MS_HAS_MPI}")