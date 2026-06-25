add_library(ms_image STATIC image.cpp)
target_include_directories(ms_image PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_image PUBLIC ms_core)
target_compile_features(ms_image PUBLIC cxx_std_23)
