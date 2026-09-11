add_library(ms_sym2 STATIC expr.cpp bridge.cpp notation.cpp notation_latex.cpp notation_mathml.cpp notation_unicode.cpp notation_interchange.cpp notation_source.cpp notation_api.cpp latex_parse.cpp)
target_include_directories(ms_sym2 PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_sym2 PUBLIC ms_core ms_bignum ms_symbolic)
target_compile_features(ms_sym2 PUBLIC cxx_std_23)
