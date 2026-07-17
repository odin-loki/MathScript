add_library(ms_fem STATIC fem.cpp)
target_include_directories(ms_fem PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_fem PUBLIC ms_core ms_linalg)
target_compile_features(ms_fem PUBLIC cxx_std_23)
