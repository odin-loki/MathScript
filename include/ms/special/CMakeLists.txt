# MathScript Special Functions Library

add_library(ms_special STATIC
    bessel.cpp
    gamma.cpp
    erf.cpp
    airy.cpp
    ellip.cpp
)

target_include_directories(ms_special PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(ms_special PUBLIC
    ms_core
    ms_runtime
)