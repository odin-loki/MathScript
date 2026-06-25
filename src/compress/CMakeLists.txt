add_library(ms_compress STATIC compress.cpp)
target_include_directories(ms_compress PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_compress PUBLIC ms_core)
target_compile_features(ms_compress PUBLIC cxx_std_23)
