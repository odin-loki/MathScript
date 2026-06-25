add_library(ms_bignum STATIC bignum.cpp)
target_include_directories(ms_bignum PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_bignum PUBLIC ms_core)
target_compile_features(ms_bignum PUBLIC cxx_std_23)
