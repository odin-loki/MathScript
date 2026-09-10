include(CTest)

# GoogleTest's EXPECT_NO_THROW expands to try/catch when GTEST_HAS_EXCEPTIONS=1.
# GCC may still define __EXCEPTIONS under -fno-exceptions; force the no-throw
# macros to execute the statement instead of using exception handling.
if(NOT MSVC)
    add_compile_definitions(GTEST_HAS_EXCEPTIONS=0)
endif()

function(add_ms_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE mathscript GTest::gtest_main)
    target_include_directories(${name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
    )
    if(MS_ENABLE_COVERAGE AND NOT MSVC)
        target_compile_options(${name} PRIVATE -fexceptions)
    endif()
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_executable(test_repl_commands
    unit/repl/test_repl_commands.cpp
    unit/repl/test_repl_commands_cfd.cpp
    unit/repl/test_repl_commands_combo.cpp
    unit/repl/test_repl_commands_compress.cpp
    unit/repl/test_repl_commands_control.cpp
    unit/repl/test_repl_commands_core.cpp
    unit/repl/test_repl_commands_cplx.cpp
    unit/repl/test_repl_commands_crypto.cpp
    unit/repl/test_repl_commands_diffgeo.cpp
    unit/repl/test_repl_commands_distributed.cpp
    unit/repl/test_repl_commands_fem.cpp
    unit/repl/test_repl_commands_fft.cpp
    unit/repl/test_repl_commands_finance.cpp
    unit/repl/test_repl_commands_frameworks.cpp
    unit/repl/test_repl_commands_geo.cpp
    unit/repl/test_repl_commands_graph.cpp
    unit/repl/test_repl_commands_image.cpp
    unit/repl/test_repl_commands_info.cpp
    unit/repl/test_repl_commands_linalg.cpp
    unit/repl/test_repl_commands_ml.cpp
    unit/repl/test_repl_commands_numthy.cpp
    unit/repl/test_repl_commands_ode.cpp
    unit/repl/test_repl_commands_pde.cpp
    unit/repl/test_repl_commands_poly.cpp
    unit/repl/test_repl_commands_prob.cpp
    unit/repl/test_repl_commands_quantum.cpp
    unit/repl/test_repl_commands_signal.cpp
    unit/repl/test_repl_commands_special.cpp
    unit/repl/test_repl_commands_stats.cpp
    unit/repl/test_repl_commands_tensorops.cpp
    unit/repl/test_repl_commands_topo.cpp
)
target_link_libraries(test_repl_commands PRIVATE mathscript GTest::gtest_main)
target_include_directories(test_repl_commands PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/unit
)
if(MS_ENABLE_COVERAGE AND NOT MSVC)
    target_compile_options(test_repl_commands PRIVATE -fexceptions)
endif()
add_test(NAME test_repl_commands COMMAND test_repl_commands)
if(MSVC)
    target_compile_options(test_repl_commands PRIVATE /bigobj)
endif()

# Generated dispatch tests for every matrix-call handler.
#
# The source list is generated too (scripts/gen_matrix_call_tests.py), so adding
# a domain directory under src/interp/matrix_calls wires its tests up without
# anyone remembering to edit this file. One executable rather than 484: these
# link the whole library and the link is what costs, especially instrumented.
include(${CMAKE_CURRENT_SOURCE_DIR}/unit/matrix_calls/generated_sources.cmake)
add_executable(test_matrix_calls ${MS_MATRIX_CALL_TEST_SOURCES})
target_link_libraries(test_matrix_calls PRIVATE mathscript GTest::gtest_main)
target_include_directories(test_matrix_calls PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    # matrix_call.hpp declares dispatch_matrix_call and is private to src/interp;
    # these tests exercise the dispatcher itself, not the REPL surface over it.
    ${CMAKE_SOURCE_DIR}/src/interp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/matrix_calls
)
if(MS_ENABLE_COVERAGE AND NOT MSVC)
    target_compile_options(test_matrix_calls PRIVATE -fexceptions)
endif()
if(MSVC)
    target_compile_options(test_matrix_calls PRIVATE /bigobj)
endif()
add_test(NAME test_matrix_calls COMMAND test_matrix_calls)

add_executable(test_mathscriptc_cli unit/interp/test_mathscriptc_cli.cpp)
target_link_libraries(test_mathscriptc_cli PRIVATE GTest::gtest_main)
target_compile_definitions(test_mathscriptc_cli PRIVATE "MATHSCRIPTC_PATH=\"$<TARGET_FILE:mathscriptc>\"")
add_dependencies(test_mathscriptc_cli mathscriptc)
add_test(NAME test_mathscriptc_cli COMMAND test_mathscriptc_cli)

# §8.3's golden transcripts. It drives the real mathscriptc rather than the library,
# because the transcript is a record of what the program prints and the program is what
# a user runs. The corpus directory is discovered at run time, so adding a transcript is
# adding two files and nothing else.
add_executable(test_repl_corpus unit/interp/test_repl_corpus.cpp)
target_link_libraries(test_repl_corpus PRIVATE GTest::gtest_main)
target_compile_definitions(test_repl_corpus PRIVATE
    "MATHSCRIPTC_PATH=\"$<TARGET_FILE:mathscriptc>\""
    "MS_REPL_CORPUS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/repl_corpus\"")
add_dependencies(test_repl_corpus mathscriptc)
add_test(NAME test_repl_corpus COMMAND test_repl_corpus)

add_executable(test_repl_cli unit/interp/test_repl_cli.cpp)
target_link_libraries(test_repl_cli PRIVATE GTest::gtest_main)
target_compile_definitions(test_repl_cli PRIVATE "MATHSCRIPT_REPL_PATH=\"$<TARGET_FILE:mathscript-repl>\"")
add_dependencies(test_repl_cli mathscript-repl)
add_test(NAME test_repl_cli COMMAND test_repl_cli)

if(TARGET mathscript-server)
    add_executable(test_server_cli unit/interp/test_server_cli.cpp)
    target_link_libraries(test_server_cli PRIVATE GTest::gtest_main)
    target_compile_definitions(test_server_cli PRIVATE "MATHSCRIPT_SERVER_PATH=\"$<TARGET_FILE:mathscript-server>\"")
    add_dependencies(test_server_cli mathscript-server)
    add_test(NAME test_server_cli COMMAND test_server_cli)
endif()

file(GLOB_RECURSE _MS_UNIT_SOURCES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "unit/*.cpp")
list(SORT _MS_UNIT_SOURCES)
set(_MS_UNIT_SKIP
    unit/plugin/test_unsafe_registry.cpp
    unit/repl/test_repl_commands.cpp
    unit/interp/test_mathscriptc_cli.cpp
    unit/interp/test_repl_corpus.cpp
    unit/interp/test_repl_cli.cpp
    unit/interp/test_server_cli.cpp
    unit/gui/test_gui_text_transforms.cpp
)
foreach(_src ${_MS_UNIT_SOURCES})
    get_filename_component(_name ${_src} NAME)
    if(_name MATCHES "^test_repl_commands")
        continue()
    endif()
    # The generated dispatch suites are linked into one test_matrix_calls
    # executable above. Left to this glob they would each also become their own
    # binary, so the same 1,377 tests would run twice and pay 29 extra links of
    # the whole library -- the cost this grouping exists to avoid.
    if(_src MATCHES "^unit/matrix_calls/")
        continue()
    endif()
    list(FIND _MS_UNIT_SKIP ${_src} _idx)
    if(NOT _idx EQUAL -1)
        continue()
    endif()
    get_filename_component(_stem ${_src} NAME_WE)
    add_ms_test(${_stem} ${_src})
endforeach()

# The unsafe registry needs src/plugin on the include path and the registry library
# on the link line, neither of which add_ms_test provides -- but it needs no LLVM,
# so it is built and run in every configuration rather than only where the AST
# plugin is available. That gate is the reason 282 lines of audit bookkeeping had
# never been tested.
add_executable(test_unsafe_registry unit/plugin/test_unsafe_registry.cpp)
target_link_libraries(test_unsafe_registry PRIVATE ms_plugin_registry GTest::gtest_main)
target_include_directories(test_unsafe_registry PRIVATE ${CMAKE_SOURCE_DIR}/src/plugin)
if(MS_ENABLE_COVERAGE AND NOT MSVC)
    target_compile_options(test_unsafe_registry PRIVATE -fexceptions)
endif()
add_test(NAME test_unsafe_registry COMMAND test_unsafe_registry)

add_ms_test(test_fuzz_stress fuzz/fuzz_stress.cpp)

if(MS_BUILD_GUI)
    find_package(Qt6 QUIET COMPONENTS Core)
    if(Qt6_FOUND)
        add_executable(test_gui_text_transforms
            unit/gui/test_gui_text_transforms.cpp
            ${CMAKE_SOURCE_DIR}/src/gui/TextTransforms.cpp)
        target_link_libraries(test_gui_text_transforms PRIVATE Qt6::Core GTest::gtest_main)
        target_include_directories(test_gui_text_transforms PRIVATE ${CMAKE_SOURCE_DIR}/src)
        add_test(NAME test_gui_text_transforms COMMAND test_gui_text_transforms)
    endif()
endif()

add_subdirectory(compliance)
add_subdirectory(numerical)
if(MS_BUILD_INTEGRATION)
    add_subdirectory(integration)
endif()

# Corpus replay: build each libFuzzer target against a plain main() that replays its
# checked-in corpus plus a fixed set of deterministic mutations. This needs no libFuzzer
# runtime, so it runs in the ordinary test suite rather than only in the 24-hour job --
# a corpus entry is a previously interesting input, and replaying one costs milliseconds.
foreach(_fuzz_target
        fuzz_special_fns fuzz_matrix_ops fuzz_repl_input fuzz_sym_parser
        fuzz_poly_ops fuzz_bignum fuzz_mpi_message)
    add_executable(replay_${_fuzz_target} fuzz/${_fuzz_target}.cpp fuzz/replay_main.cpp)
    target_link_libraries(replay_${_fuzz_target} PRIVATE mathscript)
    target_include_directories(replay_${_fuzz_target} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
    )
    if(MS_ENABLE_COVERAGE AND NOT MSVC)
        target_compile_options(replay_${_fuzz_target} PRIVATE -fexceptions)
    endif()
    add_test(NAME replay_${_fuzz_target}
             COMMAND replay_${_fuzz_target}
                     "${CMAKE_CURRENT_SOURCE_DIR}/fuzz/corpus/${_fuzz_target}" 20000)
endforeach()

if(MS_BUILD_FUZZ)
    add_subdirectory(fuzz)
endif()
