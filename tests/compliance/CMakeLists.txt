include(CTest)

# Smoke test: verifies the Clang plugin target builds when MS_BUILD_PLUGIN=ON.
add_executable(test_plugin_smoke test_plugin_smoke.cpp)
target_link_libraries(test_plugin_smoke PRIVATE GTest::gtest_main)

if(MS_BUILD_PLUGIN)
    target_compile_definitions(test_plugin_smoke PRIVATE MS_BUILD_PLUGIN=1)
    add_dependencies(test_plugin_smoke ms_plugin)
else()
    target_compile_definitions(test_plugin_smoke PRIVATE MS_BUILD_PLUGIN=0)
endif()

add_test(NAME test_plugin_smoke COMMAND test_plugin_smoke)

# Compile-fail / compile-pass rule tests (master plan §7.6).
# Each rule gets fail.cpp (must not compile with plugin) and ok.cpp (must compile).
# Uses Clang -fsyntax-only with -fplugin=ms_plugin (no link step).
function(add_compliance_test rule)
    if(NOT MS_BUILD_PLUGIN)
        message(STATUS "Skipping compliance rule '${rule}' (MS_BUILD_PLUGIN=OFF)")
        return()
    endif()

    if(MSVC)
        message(STATUS "Skipping compliance rule '${rule}' (MSVC — plugin compile tests require Clang)")
        return()
    endif()

    set(_fail_src "${CMAKE_CURRENT_SOURCE_DIR}/${rule}/fail.cpp")
    set(_ok_src "${CMAKE_CURRENT_SOURCE_DIR}/${rule}/ok.cpp")
    set(_inc "${CMAKE_SOURCE_DIR}/include")
    set(_compile_flags
        -std=c++23
        -fsyntax-only
        -fno-rtti
        -Wno-unknown-attributes
    )
    if(NOT rule STREQUAL "no_throw" AND NOT rule STREQUAL "no_catch")
        list(APPEND _compile_flags -fno-exceptions)
    endif()

    add_custom_target(compliance_${rule}_fail
        COMMAND ${CMAKE_CXX_COMPILER}
            ${_compile_flags}
            -fplugin=$<TARGET_FILE:ms_plugin>
            -I${_inc}
            -c ${_fail_src}
        DEPENDS ms_plugin
        COMMENT "Compile-fail check: ${rule}/fail.cpp"
        VERBATIM
    )
    set_target_properties(compliance_${rule}_fail PROPERTIES EXCLUDE_FROM_ALL TRUE)
    add_test(NAME compliance_${rule}_fail
        COMMAND ${CMAKE_COMMAND}
            --build ${CMAKE_BINARY_DIR}
            --target compliance_${rule}_fail
    )
    set_tests_properties(compliance_${rule}_fail PROPERTIES WILL_FAIL TRUE)

    add_custom_target(compliance_${rule}_ok
        COMMAND ${CMAKE_CXX_COMPILER}
            ${_compile_flags}
            -fplugin=$<TARGET_FILE:ms_plugin>
            -I${_inc}
            -c ${_ok_src}
        DEPENDS ms_plugin
        COMMENT "Compile-pass check: ${rule}/ok.cpp"
        VERBATIM
    )
    set_target_properties(compliance_${rule}_ok PROPERTIES EXCLUDE_FROM_ALL TRUE)
    add_test(NAME compliance_${rule}_ok
        COMMAND ${CMAKE_COMMAND}
            --build ${CMAKE_BINARY_DIR}
            --target compliance_${rule}_ok
    )
endfunction()

# Compile-only: [[ms::unsafe]] via MS_UNSAFE macro (skipped until macro exists in include/ms).
file(READ "${CMAKE_SOURCE_DIR}/include/ms/unsafe/unsafe.hpp" _ms_unsafe_hpp)
if(MS_BUILD_PLUGIN AND _ms_unsafe_hpp MATCHES "#[ \t]*define[ \t]+MS_UNSAFE")
    add_executable(compliance_unsafe_annotation EXCLUDE_FROM_ALL test_unsafe_annotation.cpp)
    target_include_directories(compliance_unsafe_annotation PRIVATE ${CMAKE_SOURCE_DIR}/include)
    set_target_properties(compliance_unsafe_annotation PROPERTIES LINKER_LANGUAGE CXX)
    add_test(NAME compliance_unsafe_annotation
        COMMAND ${CMAKE_COMMAND}
            --build ${CMAKE_BINARY_DIR}
            --target compliance_unsafe_annotation
    )
else()
    message(STATUS "Skipping compliance_unsafe_annotation (MS_UNSAFE macro not in include/ms)")
endif()

add_compliance_test(no_raw_new)
add_compliance_test(no_malloc)
add_compliance_test(no_cstyle_cast)
add_compliance_test(no_throw)
add_compliance_test(no_catch)
add_compliance_test(no_const_cast)
add_compliance_test(no_goto)
add_compliance_test(no_raw_ptr_arithmetic)
add_compliance_test(no_unsafe_reinterpret)
add_compliance_test(no_detach)
add_compliance_test(no_vla)
add_compliance_test(narrowing)
add_compliance_test(no_signed_unsigned_mix)
add_compliance_test(no_raw_thread)
add_compliance_test(no_raw_mutex_lock)
add_compliance_test(no_uninit)
add_compliance_test(no_stored_span)
add_compliance_test(no_volatile_sync)
add_compliance_test(no_owning_raw_ptr)
add_compliance_test(unused_expected)
