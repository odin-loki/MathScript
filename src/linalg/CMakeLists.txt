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
    construction.cpp
    eig.cpp
    svd.cpp
    decompositions.cpp
    matrix_funcs.cpp
    auxiliary.cpp
    iterative.cpp
    sylvester.cpp
)

target_include_directories(ms_linalg PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(ms_linalg PUBLIC ms_core ms_runtime ms_cuda)
target_compile_features(ms_linalg PUBLIC cxx_std_23)
