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
add_ms_bench(bench_linalg bench_linalg.cpp)
add_ms_bench(bench_repl bench_repl.cpp)
target_link_libraries(bench_repl PRIVATE ms_interp)
add_ms_bench(bench_special bench_special.cpp)
target_link_libraries(bench_special PRIVATE ms_special)
add_ms_bench(bench_stats bench_stats.cpp)
add_ms_bench(bench_rng_dispatch bench_rng_dispatch.cpp)
add_ms_bench(bench_simd bench_simd.cpp)
add_ms_bench(bench_signal_linalg bench_signal_linalg.cpp)
add_ms_bench(bench_ode_pde bench_ode_pde.cpp)
add_ms_bench(bench_special_memory bench_special_memory.cpp)
target_link_libraries(bench_special_memory PRIVATE ms_special)
