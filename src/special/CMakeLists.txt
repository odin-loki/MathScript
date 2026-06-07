add_library(ms_special STATIC special.cpp)
target_include_directories(ms_special PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_special PUBLIC ms_core ms_runtime)
target_compile_features(ms_special PUBLIC cxx_std_23)
