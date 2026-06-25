add_library(ms_topo STATIC topo.cpp)
target_include_directories(ms_topo PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_topo PUBLIC ms_core)
target_compile_features(ms_topo PUBLIC cxx_std_23)
