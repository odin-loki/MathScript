# Version Generation for MathScript

# Try to get git commit
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE MS_GIT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        FAIL_QUIET
    )
else()
    set(MS_GIT_COMMIT "unknown")
endif()

# Get build date
string(TIMESTAMP MS_BUILD_DATE "%Y-%m-%d" UTC)

# Generate version header
configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/version.hpp.in
    ${CMAKE_BINARY_DIR}/include/ms/version.hpp
    @ONLY
)

# Output version info
message(STATUS "MathScript Version: ${PROJECT_VERSION}")
message(STATUS "MathScript Build Date: ${MS_BUILD_DATE}")
message(STATUS "MathScript Git Commit: ${MS_GIT_COMMIT}")