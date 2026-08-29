file(GLOB_RECURSE _MS_NUMERICAL_SOURCES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "*.cpp")
list(SORT _MS_NUMERICAL_SOURCES)
foreach(_src ${_MS_NUMERICAL_SOURCES})
    get_filename_component(_stem ${_src} NAME_WE)
    string(REGEX REPLACE "^test_" "" _short ${_stem})
    add_ms_test(numerical_${_short} ${_src})
endforeach()
