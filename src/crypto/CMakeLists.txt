add_library(ms_crypto STATIC crypto.cpp ed25519.cpp)
if(MSVC)
    target_sources(ms_crypto PRIVATE curve25519_donna_msvc.cpp)
else()
    target_sources(ms_crypto PRIVATE curve25519_donna.c)
endif()
target_sources(ms_crypto PRIVATE
    ed25519/fe.c
    ed25519/ge.c
    ed25519/keypair.c
    ed25519/sc.c
    ed25519/sha512.c
    ed25519/sign.c
    ed25519/verify.c)
target_compile_definitions(ms_crypto PRIVATE ED25519_NO_SEED)
target_include_directories(ms_crypto PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ms_crypto PUBLIC ms_core)
target_compile_features(ms_crypto PUBLIC cxx_std_23)
