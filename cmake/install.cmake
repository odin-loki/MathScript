# MathScript install rules and CPack packaging.

set(_MS_STATIC_LIBS
    ms_core
    ms_runtime
    ms_linalg
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
    ms_poly
    ms_symbolic
    ms_domain
    ms_distributed
    ms_frameworks
)

foreach(_lib IN LISTS _MS_STATIC_LIBS)
    if(TARGET ${_lib})
        install(TARGETS ${_lib} ARCHIVE DESTINATION lib)
    endif()
endforeach()

install(TARGETS mathscriptc mathscript-repl mathscript-server
    RUNTIME DESTINATION bin
)

if(TARGET mathscript-gui)
    install(TARGETS mathscript-gui RUNTIME DESTINATION bin)
endif()

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ms
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

install(FILES ${CMAKE_BINARY_DIR}/include/ms/version.hpp
    DESTINATION include/ms
)

set(CPACK_PACKAGE_NAME "mathscript")
set(CPACK_PACKAGE_VENDOR "MathScript")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "MathScript HPC Computer Algebra System")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_INSTALL_DIRECTORY "MathScript")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_GENERATOR "TGZ;ZIP")

if(UNIX AND NOT APPLE)
    list(APPEND CPACK_GENERATOR "DEB;RPM")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "MathScript Contributors")
    set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
    set(CPACK_RPM_PACKAGE_LICENSE "Proprietary")
    set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
endif()

if(WIN32)
    list(APPEND CPACK_GENERATOR "NSIS")
    set(CPACK_NSIS_DISPLAY_NAME "MathScript ${PROJECT_VERSION}")
    set(CPACK_NSIS_PACKAGE_NAME "MathScript")
endif()

include(CPack)
