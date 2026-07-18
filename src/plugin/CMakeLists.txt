# MathScript Clang Plugin (stub when LLVM unavailable)

if(MS_BUILD_PLUGIN)
    find_package(LLVM REQUIRED CONFIG)
    find_package(Clang REQUIRED CONFIG)

    add_library(ms_plugin_registry STATIC unsafe_registry.cpp)
    target_include_directories(ms_plugin_registry PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    add_library(ms_plugin_rules STATIC
        rule_context.cpp
        rules/memory_rules.cpp
        rules/exception_rules.cpp
        rules/cast_rules.cpp
    )
    target_include_directories(ms_plugin_rules PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LLVM_INCLUDE_DIRS}
    )
    separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
    target_compile_definitions(ms_plugin_rules PRIVATE
        MS_BUILDING_PLUGIN=1
        ${LLVM_DEFINITIONS_LIST}
    )
    target_link_libraries(ms_plugin_rules PUBLIC ms_plugin_registry clangAST clangBasic)

    add_library(ms_plugin MODULE MsPlugin.cpp)
    target_include_directories(ms_plugin SYSTEM PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LLVM_INCLUDE_DIRS}
    )
    target_compile_definitions(ms_plugin PRIVATE
        MS_BUILDING_PLUGIN=1
        MS_PLUGIN_AUDIT_DIR="${CMAKE_BINARY_DIR}"
        ${LLVM_DEFINITIONS_LIST}
    )
    target_link_libraries(ms_plugin PRIVATE ms_plugin_rules ms_plugin_registry clangAST clangFrontend LLVM)
else()
    add_library(ms_plugin INTERFACE)
endif()
