add_library(ms_frameworks STATIC
    gria/gria.cpp
    izaac/izaac.cpp
    cypha/cypha.cpp
    cellai/cellai.cpp
    axiom/axiom.cpp
)

target_include_directories(ms_frameworks PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(ms_frameworks PUBLIC ms_core)
target_compile_features(ms_frameworks PUBLIC cxx_std_23)
