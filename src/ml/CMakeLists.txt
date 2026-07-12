add_library(ms_ml STATIC ml.cpp)
target_include_directories(ms_ml PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_ml PUBLIC ms_core ms_linalg)
target_compile_features(ms_ml PUBLIC cxx_std_23)
