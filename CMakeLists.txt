cmake_minimum_required(VERSION 3.15)

project(drift_viterbi
    VERSION 0.1.0
    DESCRIPTION "Drift-tolerance convolutional coding"
    LANGUAGES C)

option(DRIFT_VITERBI_BUILD_TESTS "Build the drift_viterbi tests" ${PROJECT_IS_TOP_LEVEL})
option(DRIFT_VITERBI_BUILD_BENCH "Build the drift_viterbi metrics and code-search harnesses" OFF)

add_subdirectory(stdlib)

add_library(drift_viterbi_interface INTERFACE)
target_include_directories(drift_viterbi_interface INTERFACE include)

# The freestanding core, compiled once as an OBJECT library so the same objects
# can be archived into both libraries below (the bare core, and the public core +
# stdlib) without compiling twice.
add_library(drift_viterbi_bare_obj OBJECT
    src/encode.c
    src/decode.c
    src/compare.c
    src/multi_decode.c
    src/multi_encode.c)
target_compile_features(drift_viterbi_bare_obj PUBLIC c_std_11)
target_compile_options(drift_viterbi_bare_obj PRIVATE -ffreestanding -fno-builtin -Wall -Wextra -Wpedantic -Werror=implicit-function-declaration)
target_link_libraries(drift_viterbi_bare_obj PUBLIC drift_viterbi_interface PRIVATE drift_viterbi_stdlib_interface)

# The freestanding core on its own, as libdrift_viterbi_bare.a (no stdlib shims -
# the consumer supplies those). $<TARGET_OBJECTS> copies the compiled objects into
# the archive rather than creating a link dependency on the object library.
add_library(drift_viterbi_bare STATIC $<TARGET_OBJECTS:drift_viterbi_bare_obj>)
target_link_libraries(drift_viterbi_bare PUBLIC drift_viterbi_interface)
target_compile_features(drift_viterbi_bare PUBLIC c_std_11)

# The public library: the core and the stdlib shims archived together into one
# self-contained libdrift_viterbi.a. Both sets of objects are copied straight in
# via $<TARGET_OBJECTS>, so there is no link dependency on the object libraries
# and every object is kept (no -Wl,--whole-archive wrapping needed). libm is
# carried through for the stdlib shims' math symbols.
add_library(drift_viterbi STATIC
    $<TARGET_OBJECTS:drift_viterbi_bare_obj>
    $<TARGET_OBJECTS:drift_viterbi_stdlib>)
target_link_libraries(drift_viterbi PUBLIC drift_viterbi_interface PRIVATE m)
target_compile_features(drift_viterbi PUBLIC c_std_11)
add_library(drift_viterbi::drift_viterbi ALIAS drift_viterbi)

if(DRIFT_VITERBI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(DRIFT_VITERBI_BUILD_BENCH)
    add_subdirectory(metrics)
    add_subdirectory(bench)
endif()

include(GNUInstallDirs)

install(TARGETS drift_viterbi drift_viterbi_bare
    EXPORT drift_viterbi-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
