add_library(ms_crypto STATIC crypto.cpp)
target_include_directories(ms_crypto PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_crypto PUBLIC ms_core)
target_compile_features(ms_crypto PUBLIC cxx_std_23)
