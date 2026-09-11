add_subdirectory(plugin)
add_subdirectory(core)
add_subdirectory(runtime)
add_subdirectory(linalg)
add_subdirectory(fft)
add_subdirectory(stats)
add_subdirectory(prob)
add_subdirectory(optim)
add_subdirectory(signal)
add_subdirectory(special)
add_subdirectory(simd)
add_subdirectory(cuda)
add_subdirectory(interp)
add_subdirectory(ode)
add_subdirectory(pde)
add_subdirectory(fem)
add_subdirectory(poly)
add_subdirectory(symbolic)
add_subdirectory(sym2)
add_subdirectory(domain)
add_subdirectory(distributed)
add_subdirectory(frameworks)
add_subdirectory(numthy)
add_subdirectory(combo)
add_subdirectory(info)
add_subdirectory(finance)
add_subdirectory(control)
add_subdirectory(graph)
add_subdirectory(cplx)
add_subdirectory(quantum)
add_subdirectory(geo)
add_subdirectory(diffgeo)
add_subdirectory(topo)
add_subdirectory(tensorops)
add_subdirectory(ml)
add_subdirectory(image)
add_subdirectory(compress)
add_subdirectory(cfd)
add_subdirectory(bignum)
add_subdirectory(crypto)
add_subdirectory(gui)
add_subdirectory(exe)

set(_MS_LIBS
    ms_core
    ms_linalg
    ms_runtime
    ms_fft
    ms_stats
    ms_prob
    ms_optim
    ms_signal
    ms_special
    ms_simd
    ms_cuda
    ms_interp
    ms_ode
    ms_pde
    ms_fem
    ms_poly
    ms_symbolic
    ms_sym2
    ms_domain
    ms_distributed
    ms_frameworks
    ms_numthy
    ms_combo
    ms_info
    ms_finance
    ms_control
    ms_graph
    ms_cplx
    ms_quantum
    ms_geo
    ms_diffgeo
    ms_topo
    ms_tensorops
    ms_ml
    ms_image
    ms_compress
    ms_cfd
    ms_bignum
    ms_crypto
)

if(MS_ENABLE_COVERAGE AND NOT MSVC)
    foreach(_lib IN LISTS _MS_LIBS)
        if(TARGET ${_lib})
            target_compile_options(${_lib} PRIVATE --coverage -fprofile-arcs -ftest-coverage -O0 -g)
        endif()
    endforeach()
endif()

add_library(mathscript INTERFACE)
if(MS_LINK_TESTS_SHARED AND NOT MSVC)
    # Build the bundle from the module objects directly rather than by linking the
    # archives with WHOLE_ARCHIVE.
    #
    # The previous form did the latter and needed --allow-multiple-definition to
    # link at all. That flag was not hiding two different implementations of one
    # symbol -- it was hiding the fact that every archive reached the link line
    # TWICE, once as an explicit WHOLE_ARCHIVE item and again as a transitive
    # PUBLIC dependency of a sibling (every module links ms_core). With
    # LINK_LIBRARY_OVERRIDE pinning WHOLE_ARCHIVE onto both occurrences, each
    # object was pulled in twice and every symbol in the tree was defined twice:
    # 20 reported duplicates, all byte-identical copies of themselves.
    #
    # Suppressing that also suppressed the diagnostic, so a genuine duplicate --
    # two different functions with one mangled name, where the linker's arbitrary
    # choice would change behaviour -- would have linked just as quietly.
    #
    # $<TARGET_OBJECTS:> names each module's objects exactly once, so there is
    # nothing to deduplicate and the flag is gone. The libraries are still linked,
    # for their INTERFACE include directories and their external dependencies, but
    # as ordinary items that contribute no objects of their own.
    set(_ms_bundle_objs)
    foreach(_lib IN LISTS _MS_LIBS)
        if(TARGET ${_lib})
            list(APPEND _ms_bundle_objs "$<TARGET_OBJECTS:${_lib}>")
        endif()
    endforeach()
    add_library(ms_bundle SHARED ms_bundle.cpp ${_ms_bundle_objs})
    foreach(_lib IN LISTS _MS_LIBS)
        if(TARGET ${_lib})
            target_link_libraries(ms_bundle PRIVATE $<LINK_ONLY:${_lib}>)
        endif()
    endforeach()
    target_include_directories(ms_bundle PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
    )
    if(MS_ENABLE_COVERAGE)
        target_link_options(ms_bundle PRIVATE --coverage)
    endif()
    target_link_libraries(mathscript INTERFACE ms_bundle)
else()
    target_link_libraries(mathscript INTERFACE ${_MS_LIBS})
endif()
target_include_directories(mathscript INTERFACE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)
