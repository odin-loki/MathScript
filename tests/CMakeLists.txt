include(CTest)

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

add_executable(test_mathscriptc_cli unit/interp/test_mathscriptc_cli.cpp)
target_link_libraries(test_mathscriptc_cli PRIVATE GTest::gtest_main)
target_compile_definitions(test_mathscriptc_cli PRIVATE "MATHSCRIPTC_PATH=\"$<TARGET_FILE:mathscriptc>\"")
add_dependencies(test_mathscriptc_cli mathscriptc)
add_test(NAME test_mathscriptc_cli COMMAND test_mathscriptc_cli)

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
    unit/repl/test_repl_commands.cpp
    unit/interp/test_mathscriptc_cli.cpp
    unit/interp/test_repl_cli.cpp
    unit/interp/test_server_cli.cpp
    unit/gui/test_gui_text_transforms.cpp
)
foreach(_src ${_MS_UNIT_SOURCES})
    get_filename_component(_name ${_src} NAME)
    if(_name MATCHES "^test_repl_commands")
        continue()
    endif()
    list(FIND _MS_UNIT_SKIP ${_src} _idx)
    if(NOT _idx EQUAL -1)
        continue()
    endif()
    get_filename_component(_stem ${_src} NAME_WE)
    add_ms_test(${_stem} ${_src})
endforeach()

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
add_subdirectory(integration)

if(MS_BUILD_FUZZ)
    add_subdirectory(fuzz)
endif()
