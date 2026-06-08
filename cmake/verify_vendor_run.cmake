# Run vendor checksum verification (cmake -P mode).
# Invoked from verify_vendor.cmake; repo root is derived from this file's location.

get_filename_component(MS_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

set(_MS_CHECKSUMS_FILE "${MS_SOURCE_DIR}/vendor/CHECKSUMS.sha256")

if(NOT EXISTS "${_MS_CHECKSUMS_FILE}")
    message(FATAL_ERROR "Missing vendor/CHECKSUMS.sha256")
endif()

file(STRINGS "${_MS_CHECKSUMS_FILE}" _lines)

set(_count 0)
foreach(_line IN LISTS _lines)
    string(STRIP "${_line}" _line)
    if(_line STREQUAL "" OR _line MATCHES "^#")
        continue()
    endif()
    if(NOT _line MATCHES "^([0-9a-fA-F]{64})[[:space:]]+(.*)$")
        message(FATAL_ERROR "Invalid checksum line in vendor/CHECKSUMS.sha256: ${_line}")
    endif()
    set(_expected "${CMAKE_MATCH_1}")
    set(_relpath "${CMAKE_MATCH_2}")
    string(REGEX REPLACE "^\\./" "" _relpath "${_relpath}")
    set(_filepath "${MS_SOURCE_DIR}/${_relpath}")
    if(NOT EXISTS "${_filepath}")
        message(FATAL_ERROR "Missing vendored file: ${_relpath}")
    endif()
    file(SHA256 "${_filepath}" _actual)
    string(TOLOWER "${_expected}" _expected_lower)
    if(NOT _actual STREQUAL _expected_lower)
        message(FATAL_ERROR
            "Vendor integrity check failed for ${_relpath}\n"
            "  expected: ${_expected_lower}\n"
            "  actual:   ${_actual}")
    endif()
    math(EXPR _count "${_count} + 1")
endforeach()

if(_count EQUAL 0)
    message(STATUS "no vendored sources")
else()
    message(STATUS "Vendor integrity check OK (${_count} file(s))")
endif()
