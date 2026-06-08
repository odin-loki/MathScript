if(MS_BUILD_GUI)
    find_package(Qt6 QUIET COMPONENTS Widgets OpenGLWidgets OpenGL)
    if(Qt6_FOUND)
        set(CMAKE_AUTOMOC ON)
        add_library(ms_gui STATIC
            MainWindow.cpp MainWindow.hpp
            PlotWidget.cpp PlotWidget.hpp
            PlotSurfWidget.cpp PlotSurfWidget.hpp)
        target_include_directories(ms_gui PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
        target_link_libraries(ms_gui PUBLIC Qt6::Widgets Qt6::OpenGLWidgets Qt6::OpenGL ms_interp)
        target_compile_features(ms_gui PUBLIC cxx_std_23)

        add_executable(mathscript-gui mathscript_gui.cpp)
        target_link_libraries(mathscript-gui PRIVATE ms_gui)
        target_include_directories(mathscript-gui PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
        set_target_properties(mathscript-gui PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
            WIN32_EXECUTABLE TRUE
        )
        message(STATUS "MathScript Qt6 IDE enabled")
    else()
        message(STATUS "MathScript Qt6 not found; GUI target skipped")
    endif()
endif()
