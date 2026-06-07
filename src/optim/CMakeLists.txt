add_library(ms_optim STATIC optim.cpp)
target_include_directories(ms_optim PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_optim PUBLIC ms_core)
target_compile_features(ms_optim PUBLIC cxx_std_23)
