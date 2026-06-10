# MathScript Executables

add_executable(mathscriptc mathscriptc.cpp)
target_link_libraries(mathscriptc PRIVATE mathscript ms_interp)
target_include_directories(mathscriptc PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)

add_executable(mathscript-repl repl.cpp)
target_link_libraries(mathscript-repl PRIVATE mathscript ms_interp)

add_executable(mathscript-server mathscript_server.cpp)
target_link_libraries(mathscript-server PRIVATE ms_distributed)
target_include_directories(mathscript-server PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
target_include_directories(mathscript-repl PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)

set_target_properties(mathscriptc mathscript-repl mathscript-server PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
