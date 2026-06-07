add_library(ms_pde STATIC pde.cpp)
target_include_directories(ms_pde PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_pde PUBLIC ms_core)
target_compile_features(ms_pde PUBLIC cxx_std_23)
