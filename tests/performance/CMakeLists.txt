function(add_ms_bench name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE mathscript benchmark::benchmark benchmark::benchmark_main)
    target_include_directories(${name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
    )
endfunction()

add_ms_bench(bench_matmul bench_matmul.cpp)
add_ms_bench(bench_fft bench_fft.cpp)
