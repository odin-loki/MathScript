# MathScript Clang Plugin (stub when LLVM unavailable)

if(MS_BUILD_PLUGIN)
    find_package(LLVM REQUIRED CONFIG)
    find_package(Clang REQUIRED CONFIG)
    add_library(ms_plugin MODULE MsPlugin.cpp)
    separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
    target_include_directories(ms_plugin SYSTEM PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LLVM_INCLUDE_DIRS}
    )
    target_compile_definitions(ms_plugin PRIVATE MS_BUILDING_PLUGIN=1 ${LLVM_DEFINITIONS_LIST})
    target_link_libraries(ms_plugin PUBLIC clangAST clangFrontend LLVM)
else()
    add_library(ms_plugin INTERFACE)
endif()
