cmake_minimum_required(VERSION 3.15)

project(drift_viterbi
    VERSION 0.1.0
    DESCRIPTION "Drift-tolerance convolutional coding"
    LANGUAGES C)

option(DRIFT_VITERBI_BUILD_TESTS "Build the drift_viterbi tests" ${PROJECT_IS_TOP_LEVEL})
option(DRIFT_VITERBI_BUILD_BENCH "Build the drift_viterbi metrics harness" OFF)

add_subdirectory(stdlib)

add_library(drift_viterbi_interface INTERFACE)
target_include_directories(drift_viterbi_interface INTERFACE include)

add_library(drift_viterbi_bare STATIC
    src/encode.c
    src/decode.c
    src/compare.c
    src/multi_decode.c
    src/multi_encode.c)

target_include_directories(drift_viterbi_bare PUBLIC include)
target_compile_features(drift_viterbi_bare PUBLIC c_std_11)
target_compile_options(drift_viterbi_bare PRIVATE -ffreestanding -fno-builtin -Wall -Wextra -Wpedantic -Werror=implicit-function-declaration)
target_link_libraries(drift_viterbi_bare PUBLIC drift_viterbi_interface PRIVATE drift_viterbi_stdlib_interface)
target_link_options(drift_viterbi_bare PRIVATE -nostdlib)

add_library(drift_viterbi STATIC src/empty.c)
target_link_libraries(drift_viterbi 
    PUBLIC drift_viterbi_interface 
    PRIVATE -Wl,-whole-archive drift_viterbi_bare drift_viterbi_stdlib -Wl,-no-whole-archive)
add_library(drift_viterbi::drift_viterbi ALIAS drift_viterbi)

if(DRIFT_VITERBI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(DRIFT_VITERBI_BUILD_BENCH)
    add_subdirectory(bench)
endif()

include(GNUInstallDirs)

install(TARGETS drift_viterbi
    EXPORT drift_viterbi-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
