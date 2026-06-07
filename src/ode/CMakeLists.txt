add_library(ms_ode STATIC ode.cpp)
target_include_directories(ms_ode PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_ode PUBLIC ms_core)
target_compile_features(ms_ode PUBLIC cxx_std_23)
