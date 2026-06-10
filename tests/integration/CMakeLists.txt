add_ms_test(integration_repl_pipeline test_repl_pipeline.cpp)
add_ms_test(integration_jit_repl_parity test_jit_repl_parity.cpp)

add_executable(integration_mathscriptc_pipeline test_mathscriptc_pipeline.cpp)
target_link_libraries(integration_mathscriptc_pipeline PRIVATE mathscript GTest::gtest_main)
target_include_directories(integration_mathscriptc_pipeline PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)
target_compile_definitions(integration_mathscriptc_pipeline PRIVATE
    "MATHSCRIPTC_PATH=\"$<TARGET_FILE:mathscriptc>\""
)
add_dependencies(integration_mathscriptc_pipeline mathscriptc)
add_test(NAME integration_mathscriptc_pipeline COMMAND integration_mathscriptc_pipeline)
