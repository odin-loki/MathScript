if(MS_ENABLE_COVERAGE)
    if(MS_COMPILER STREQUAL "clang" OR MS_COMPILER STREQUAL "gcc")
        add_compile_options(-O0 -g --coverage -fprofile-arcs -ftest-coverage)
        add_link_options(--coverage)
        message(STATUS "Coverage instrumentation enabled (gcov/llvm-cov)")
        if(MS_BUILD_TESTS)
            add_custom_target(coverage_report
                COMMAND ${CMAKE_COMMAND} -E env MS_COVERAGE_MIN=$ENV{MS_COVERAGE_MIN}
                    bash ${CMAKE_SOURCE_DIR}/scripts/coverage_report.sh ${CMAKE_BINARY_DIR}
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                COMMENT "Generate lcov report from instrumented build"
                VERBATIM
            )
        endif()
    elseif(MSVC)
        message(STATUS "Coverage instrumentation: use OpenCppCoverage or clang-cl with -fprofile-instr-generate on MSVC builds")
    endif()
endif()
