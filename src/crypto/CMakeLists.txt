add_library(ms_crypto STATIC crypto.cpp)
if(MSVC)
    target_sources(ms_crypto PRIVATE curve25519_donna_msvc.cpp)
else()
    target_sources(ms_crypto PRIVATE curve25519_donna.c)
endif()
target_include_directories(ms_crypto PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_crypto PUBLIC ms_core)
target_compile_features(ms_crypto PUBLIC cxx_std_23)
