# Vendor SHA-256 integrity verification (master plan §13.8)

set(_MS_CHECKSUMS_FILE "${CMAKE_SOURCE_DIR}/vendor/CHECKSUMS.sha256")
set(_MS_VENDOR_HAS_CHECKSUMS OFF)

if(EXISTS "${_MS_CHECKSUMS_FILE}")
    file(STRINGS "${_MS_CHECKSUMS_FILE}" _MS_CHECKSUMS_LINES)
    foreach(_line IN LISTS _MS_CHECKSUMS_LINES)
        string(STRIP "${_line}" _line)
        if(_line MATCHES "^[0-9a-fA-F]{64}[[:space:]]+")
            set(_MS_VENDOR_HAS_CHECKSUMS ON)
            break()
        endif()
    endforeach()
endif()

if(_MS_VENDOR_HAS_CHECKSUMS)
    option(MS_VERIFY_VENDOR "Verify vendored source SHA-256 checksums before building" ON)
else()
    option(MS_VERIFY_VENDOR "Verify vendored source SHA-256 checksums before building" OFF)
endif()

set(_MS_VERIFY_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/verify_vendor.sh")
set(_MS_VERIFY_RUN "${CMAKE_SOURCE_DIR}/cmake/verify_vendor_run.cmake")
set(_MS_BASH_EXECUTABLE "")

if(WIN32)
    find_program(GIT_EXECUTABLE git)
    if(GIT_EXECUTABLE)
        get_filename_component(_MS_GIT_ROOT "${GIT_EXECUTABLE}" DIRECTORY)
        if(_MS_GIT_ROOT MATCHES "[/\\\\]cmd$")
            get_filename_component(_MS_GIT_ROOT "${_MS_GIT_ROOT}" DIRECTORY)
        endif()
        find_program(_MS_BASH_EXECUTABLE
            NAMES bash.exe bash
            PATHS "${_MS_GIT_ROOT}/bin"
            NO_DEFAULT_PATH
        )
    endif()
endif()

if(NOT _MS_BASH_EXECUTABLE)
    find_program(_MS_BASH_EXECUTABLE bash)
    if(_MS_BASH_EXECUTABLE MATCHES "[Ww]indows[Aa]pps")
        unset(_MS_BASH_EXECUTABLE)
    endif()
endif()

if(MS_VERIFY_VENDOR)
    if(_MS_BASH_EXECUTABLE AND EXISTS "${_MS_VERIFY_SCRIPT}")
        add_custom_target(verify_vendor ALL
            COMMAND "${_MS_BASH_EXECUTABLE}" "${_MS_VERIFY_SCRIPT}"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            COMMENT "Verifying vendored source SHA-256 checksums"
            VERBATIM
        )
    else()
        add_custom_target(verify_vendor ALL
            COMMAND "${CMAKE_COMMAND}" -P "${_MS_VERIFY_RUN}"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            COMMENT "Verifying vendored source SHA-256 checksums"
            VERBATIM
        )
    endif()
else()
    add_custom_target(verify_vendor
        COMMAND "${CMAKE_COMMAND}" -E echo "MS_VERIFY_VENDOR is OFF; skipping vendor checksum verification"
        COMMENT "Vendor verification skipped (MS_VERIFY_VENDOR=OFF)"
        VERBATIM
    )
endif()
