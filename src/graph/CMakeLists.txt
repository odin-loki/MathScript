add_library(ms_graph STATIC graph.cpp)
target_include_directories(ms_graph PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_graph PUBLIC ms_core)
target_compile_features(ms_graph PUBLIC cxx_std_23)
