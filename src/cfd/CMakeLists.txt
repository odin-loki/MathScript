add_library(ms_cfd STATIC cfd.cpp)
target_include_directories(ms_cfd PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_cfd PUBLIC ms_core)
target_compile_features(ms_cfd PUBLIC cxx_std_23)
