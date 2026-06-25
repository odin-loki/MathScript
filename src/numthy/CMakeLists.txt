add_library(ms_numthy STATIC numthy.cpp)
target_include_directories(ms_numthy PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_numthy PUBLIC ms_core)
target_compile_features(ms_numthy PUBLIC cxx_std_23)
