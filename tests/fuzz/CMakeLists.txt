include(CheckCXXCompilerFlag)

function(add_ms_fuzz_target name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE mathscript)
    target_include_directories(${name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
    )

    if(MSVC)
        target_compile_options(${name} PRIVATE /fsanitize=fuzzer)
        target_link_options(${name} PRIVATE /fsanitize=fuzzer)
    else()
        target_link_options(${name} PRIVATE -fsanitize=fuzzer)
    endif()
endfunction()

check_cxx_compiler_flag("-fsanitize=fuzzer" MS_CXX_SUPPORTS_LIBFUZZER)
if(MS_CXX_SUPPORTS_LIBFUZZER OR MS_COMPILER STREQUAL "clang")
    add_ms_fuzz_target(fuzz_special_fns fuzz_special_fns.cpp)
else()
    message(STATUS "libFuzzer unsupported by ${CMAKE_CXX_COMPILER_ID}; fuzz targets skipped")
endif()
