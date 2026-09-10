# Integration tests, one executable per mathematical domain.
#
# These used to be one executable per source file: 573 binaries, each linking the
# whole library. Linking, not compiling, is what dominates a test build, and on an
# instrumented build it dominates it badly enough that measuring coverage was a
# nightly event rather than something anyone did before pushing. Grouping by
# domain runs exactly the same tests through 29 links instead of 573.
#
# The prerequisite was test names unique within a domain. TEST(A, B) expands to a
# class A_B_Test whose member functions are implicitly inline, so two files in one
# binary declaring the same pair link with no diagnostic and one body silently
# replaces the other. 71 pairs collided, 29 of them with different bodies; those
# were disambiguated before this change, and the guard below keeps them that way.
#
# Target names stay short so MSVC object paths fit CMAKE_OBJECT_PATH_MAX.

file(GLOB _MS_INTEGRATION_DIRS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/*")
list(SORT _MS_INTEGRATION_DIRS)

set(_MS_INTEGRATION_TARGETS "")
foreach(_dir ${_MS_INTEGRATION_DIRS})
    if(NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${_dir}")
        continue()
    endif()

    file(GLOB_RECURSE _srcs RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/${_dir}/*.cpp")
    if(NOT _srcs)
        continue()
    endif()
    list(SORT _srcs)

    string(REPLACE "/" "_" _name "int_${_dir}")
    add_executable(${_name} ${_srcs})
    target_link_libraries(${_name} PRIVATE mathscript GTest::gtest_main)
    target_include_directories(${_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
    )
    if(MS_ENABLE_COVERAGE AND NOT MSVC)
        target_compile_options(${_name} PRIVATE -fexceptions)
    endif()
    if(MSVC)
        # Many translation units per target; some domains have 85.
        target_compile_options(${_name} PRIVATE /bigobj)
    endif()
    add_test(NAME ${_name} COMMAND ${_name})
    list(APPEND _MS_INTEGRATION_TARGETS ${_name})

    # One test drives the compiler binary rather than the library, so its whole
    # domain target needs to know where that binary landed and to wait for it.
    set(_needs_compiler FALSE)
    foreach(_s ${_srcs})
        if(_s MATCHES "test_mathscriptc_pipeline\\.cpp$")
            set(_needs_compiler TRUE)
        endif()
    endforeach()
    if(_needs_compiler)
        target_compile_definitions(${_name} PRIVATE
            "MATHSCRIPTC_PATH=\"$<TARGET_FILE:mathscriptc>\"")
        add_dependencies(${_name} mathscriptc)
    endif()
endforeach()

list(LENGTH _MS_INTEGRATION_TARGETS _MS_INTEGRATION_COUNT)
message(STATUS "Integration tests: ${_MS_INTEGRATION_COUNT} domain targets")
