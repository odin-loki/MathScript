# MathScript Core Library

add_library(ms_core STATIC
    tensor.cpp
    sparse.cpp
    sym.cpp
    scalar.cpp
    error_format.cpp
    rng.cpp
)

target_include_directories(ms_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)

target_compile_features(ms_core PUBLIC cxx_std_23)
