add_library(ms_fft STATIC fft.cpp)
target_include_directories(ms_fft PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_fft PUBLIC ms_core ms_runtime ms_cuda)
target_compile_features(ms_fft PUBLIC cxx_std_23)
