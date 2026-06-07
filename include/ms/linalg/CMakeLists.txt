# MathScript Linear Algebra Library

add_library(ms_linalg STATIC
    matmul.cpp
    lu.cpp
    qr.cpp
    chol.cpp
    solve.cpp
    expm.cpp
    trace.cpp
    det.cpp
    norm.cpp
    eye.cpp
    zeros.cpp
    ones.cpp
    transpose.cpp
)

target_include_directories(ms_linalg PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(ms_linalg PUBLIC
    ms_core
    ms_runtime
)