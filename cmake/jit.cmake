# Optional LLVM ORC JIT backend (Phase 10). Default OFF; enable with -DMS_BUILD_JIT=ON.

if(MS_BUILD_JIT)
    find_package(LLVM REQUIRED CONFIG)
    message(STATUS "MathScript JIT: LLVM ${LLVM_PACKAGE_VERSION}")

    llvm_map_components_to_libnames(_MS_LLVM_JIT_LIBS orcjit native)

    function(ms_apply_jit_llvm target)
        target_compile_definitions(${target} PRIVATE MS_JIT_LLVM=1)
        separate_arguments(_MS_LLVM_DEFINITIONS NATIVE_COMMAND ${LLVM_DEFINITIONS})
        target_compile_definitions(${target} PRIVATE ${_MS_LLVM_DEFINITIONS})
        target_include_directories(${target} SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
        target_link_libraries(${target} PRIVATE ${_MS_LLVM_JIT_LIBS})
    endfunction()
else()
    function(ms_apply_jit_llvm target)
    endfunction()
endif()
