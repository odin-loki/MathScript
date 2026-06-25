add_library(ms_control STATIC control.cpp)
target_include_directories(ms_control PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_control PUBLIC ms_core)
target_compile_features(ms_control PUBLIC cxx_std_23)
