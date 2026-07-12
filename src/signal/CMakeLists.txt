add_library(ms_signal STATIC signal.cpp)
target_include_directories(ms_signal PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_signal PUBLIC ms_core ms_fft ms_linalg)
target_compile_features(ms_signal PUBLIC cxx_std_23)
