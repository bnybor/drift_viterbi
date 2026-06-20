cmake_minimum_required(VERSION 3.15)

project(drift_viterbi
    VERSION 0.1.0
    DESCRIPTION "A C library"
    LANGUAGES C)

# ---- Options ----------------------------------------------------------------

option(DRIFT_VITERBI_BUILD_TESTS "Build the drift_viterbi tests" ${PROJECT_IS_TOP_LEVEL})
option(DRIFT_VITERBI_BUILD_BENCH "Build the drift_viterbi metrics harness" OFF)

# ---- Library ----------------------------------------------------------------

add_library(drift_viterbi
    src/encode.c
    src/decode.c
    src/compare.c)

# Namespaced alias so the target can be used the same way whether this project
# is consumed via add_subdirectory() or find_package().
add_library(drift_viterbi::drift_viterbi ALIAS drift_viterbi)

target_include_directories(drift_viterbi
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

target_compile_features(drift_viterbi PUBLIC c_std_11)

# log() from <math.h> needs libm on most Unix toolchains.
if(NOT MSVC)
    target_link_libraries(drift_viterbi PUBLIC m)
endif()

set_target_properties(drift_viterbi PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})

if(MSVC)
    target_compile_options(drift_viterbi PRIVATE /W4)
else()
    target_compile_options(drift_viterbi PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---- Install ----------------------------------------------------------------

include(GNUInstallDirs)

install(TARGETS drift_viterbi
    EXPORT drift_viterbi-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# ---- Tests ------------------------------------------------------------------

if(DRIFT_VITERBI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---- Benchmarks -------------------------------------------------------------

if(DRIFT_VITERBI_BUILD_BENCH)
    add_subdirectory(bench)
endif()
