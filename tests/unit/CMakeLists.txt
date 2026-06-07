# MathScript Unit Tests

function(add_ms_test name source)
    add_executable(test_${name} ${source})
    target_link_libraries(test_${name} PRIVATE ms_core ms_linalg)
    add_test(NAME test_${name} COMMAND test_${name})
endfunction()

# Matrix operation tests
add_ms_test(matmul        src/core/operations.cpp)
add_ms_test(lu            src/core/operations.cpp)
add_ms_test(qr            src/core/operations.cpp)
add_ms_test(chol          src/core/operations.cpp)
add_ms_test(solve         src/core/operations.cpp)
add_ms_test(expm          src/core/operations.cpp)
add_ms_test(trace         src/core/operations.cpp)
add_ms_test(det           src/core/operations.cpp)
add_ms_test(norm          src/core/operations.cpp)
add_ms_test(eye           src/core/operations.cpp)
add_ms_test(zeros         src/core/operations.cpp)
add_ms_test(ones          src/core/operations.cpp)
add_ms_test(transpose     src/core/operations.cpp)