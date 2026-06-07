add_library(ms_interp STATIC repl_engine.cpp)
target_include_directories(ms_interp PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_interp PUBLIC ms_core ms_linalg ms_fft ms_runtime ms_simd ms_distributed ms_frameworks)
target_compile_features(ms_interp PUBLIC cxx_std_23)
