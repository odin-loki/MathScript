set(_MS_INTERP_SOURCES repl_engine.cpp plot_console.cpp jit_backend_factory.cpp jit_orc_stub.cpp)

if(MS_BUILD_JIT)
    list(APPEND _MS_INTERP_SOURCES jit_orc_llvm.cpp)
endif()

add_library(ms_interp STATIC ${_MS_INTERP_SOURCES})
target_include_directories(ms_interp PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_interp PUBLIC
    ms_core ms_linalg ms_fft ms_runtime ms_simd ms_distributed ms_frameworks ms_special
)
target_compile_features(ms_interp PUBLIC cxx_std_23)
if(MS_ENABLE_COVERAGE AND NOT MSVC)
    target_compile_options(ms_interp PRIVATE -fexceptions)
endif()

if(MS_BUILD_JIT)
    ms_apply_jit_llvm(ms_interp)
endif()
