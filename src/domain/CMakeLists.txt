add_library(ms_domain STATIC domain.cpp)
target_include_directories(ms_domain PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_domain PUBLIC ms_core)
target_compile_features(ms_domain PUBLIC cxx_std_23)
