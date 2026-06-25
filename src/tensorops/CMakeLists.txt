add_library(ms_tensorops STATIC tensorops.cpp)
target_include_directories(ms_tensorops PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_tensorops PUBLIC ms_core)
target_compile_features(ms_tensorops PUBLIC cxx_std_23)
