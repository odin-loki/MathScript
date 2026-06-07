add_library(ms_stats STATIC stats.cpp)
target_include_directories(ms_stats PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_stats PUBLIC ms_core)
target_compile_features(ms_stats PUBLIC cxx_std_23)
