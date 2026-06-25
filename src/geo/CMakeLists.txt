add_library(ms_geo STATIC geo.cpp)
target_include_directories(ms_geo PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_geo PUBLIC ms_core)
target_compile_features(ms_geo PUBLIC cxx_std_23)
