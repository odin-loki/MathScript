add_library(ms_finance STATIC finance.cpp)
target_include_directories(ms_finance PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
target_link_libraries(ms_finance PUBLIC ms_core)
target_compile_features(ms_finance PUBLIC cxx_std_23)
