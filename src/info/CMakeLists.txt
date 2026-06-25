add_library(ms_info STATIC info.cpp)
target_include_directories(ms_info PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_info PUBLIC ms_core)
target_compile_features(ms_info PUBLIC cxx_std_23)
