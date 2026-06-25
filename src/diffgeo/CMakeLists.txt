add_library(ms_diffgeo STATIC diffgeo.cpp)
target_include_directories(ms_diffgeo PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_diffgeo PUBLIC ms_core)
target_compile_features(ms_diffgeo PUBLIC cxx_std_23)
