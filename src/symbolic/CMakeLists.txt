add_library(ms_symbolic STATIC symbolic.cpp)
target_include_directories(ms_symbolic PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_symbolic PUBLIC ms_core)
target_compile_features(ms_symbolic PUBLIC cxx_std_23)
