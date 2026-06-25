add_library(ms_cplx STATIC cplx.cpp)
target_include_directories(ms_cplx PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_cplx PUBLIC ms_core)
target_compile_features(ms_cplx PUBLIC cxx_std_23)
