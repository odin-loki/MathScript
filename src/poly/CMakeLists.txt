add_library(ms_poly STATIC poly.cpp)
target_include_directories(ms_poly PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_poly PUBLIC ms_core ms_linalg ms_simd)
target_compile_features(ms_poly PUBLIC cxx_std_23)
