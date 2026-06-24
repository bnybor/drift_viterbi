cmake_minimum_required(VERSION 3.15)

project(drift_viterbi
    VERSION 0.1.0
    DESCRIPTION "Drift-tolerance convolutional coding"
    LANGUAGES C)

option(DRIFT_VITERBI_BUILD_TESTS "Build the drift_viterbi tests" ${PROJECT_IS_TOP_LEVEL})
option(DRIFT_VITERBI_BUILD_BENCH "Build the drift_viterbi metrics and code-search harnesses" OFF)

# Size-focused flags for the MinSizeRel config (applied to the core and stdlib
# below, since this precedes their targets). The library is plain C with no
# exceptions, so the unwind tables (.eh_frame, ~17% of the object) and the
# compiler .ident string are dead weight; dropping them shrinks the archive
# notably. The trade-off is that debuggers/profilers cannot unwind through these
# frames - acceptable for a minimum-size release build.
add_compile_options($<$<CONFIG:MinSizeRel>:-fno-asynchronous-unwind-tables>
                    $<$<CONFIG:MinSizeRel>:-fno-unwind-tables>
                    $<$<CONFIG:MinSizeRel>:-fno-ident>)

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

# Only the public API - the symbols declared in the installed include/ headers -
# should be exported from the archives. The decoder shares internal helpers across
# translation units (dv_decode_step, dv_trellis_*, dv_decode_ctx_*), and the full
# library also bundles the stdlib shims (dv_malloc, dv_memcpy, dv_log, ...); both
# are internal plumbing that would otherwise land in the archive's symbol table.
# Because they are referenced across object boundaries, hiding them needs a partial
# link: combine an archive's objects into one relocatable object, then localize
# every *defined* symbol except the public API. Undefined references are left
# alone, so the bare archive still asks the consumer to supply the shims and the
# full archive still resolves libc at the consumer's final link.
set(_dv_public_symbols
    drift_viterbi_version
    dv_code_create dv_code_create_standard dv_code_destroy
    dv_code_encode dv_code_encode_flush dv_code_k dv_code_n
    dv_compare dv_compare_max_len dv_compare_min_len
    dv_detect dv_detect_max_len dv_detect_min_len dv_detect_sequential
    dv_multi_create dv_multi_decode dv_multi_decode_flush dv_multi_destroy
    dv_multi_encode dv_multi_encode_create dv_multi_encode_destroy
    dv_multi_encode_flush
    dv_stream_decode dv_stream_decode_flush
    dv_stream_decoder_create dv_stream_decoder_destroy)
string(REPLACE ";" "\n" _dv_public_symbols_text "${_dv_public_symbols}")
set(_dv_keep_file ${CMAKE_CURRENT_BINARY_DIR}/dv_public_symbols.txt)
file(WRITE ${_dv_keep_file} "${_dv_public_symbols_text}\n")

# Build STATIC library `name` from the given OBJECT-library targets, exporting only
# the public API (see above). Falls back to a plain multi-object archive where the
# binutils ld -r / objcopy are unavailable - that still works, it just leaves the
# internals exported.
function(dv_add_archive name)
    set(_obj_exprs "")
    foreach(_lib IN LISTS ARGN)
        list(APPEND _obj_exprs $<TARGET_OBJECTS:${_lib}>)
    endforeach()
    if(CMAKE_LINKER AND CMAKE_OBJCOPY)
        set(_merged ${CMAKE_CURRENT_BINARY_DIR}/${name}_merged.o)
        # In MinSizeRel, follow the localization with a strip of the local symbol
        # names that survived the partial link only because relocations referenced
        # them - they are dead weight in the archive. A separate pass so it runs
        # after, not interleaved with, the localization.
        set(_strip "")
        if(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
            set(_strip COMMAND ${CMAKE_OBJCOPY} --strip-unneeded ${_merged})
        endif()
        add_custom_command(
            OUTPUT ${_merged}
            COMMAND ${CMAKE_LINKER} -r -o ${_merged} ${_obj_exprs}
            COMMAND ${CMAKE_OBJCOPY} --keep-global-symbols=${_dv_keep_file} ${_merged}
            ${_strip}
            DEPENDS ${ARGN} ${_dv_keep_file} ${_obj_exprs}
            COMMENT "Linking and localizing internal symbols for lib${name}.a"
            COMMAND_EXPAND_LISTS VERBATIM)
        set_source_files_properties(${_merged} PROPERTIES
            EXTERNAL_OBJECT TRUE GENERATED TRUE)
        add_library(${name} STATIC ${_merged})
        set_target_properties(${name} PROPERTIES LINKER_LANGUAGE C)
    else()
        add_library(${name} STATIC ${_obj_exprs})
    endif()
endfunction()

# The freestanding core on its own, as libdrift_viterbi_bare.a (no stdlib shims -
# the consumer supplies those, so they stay undefined here).
dv_add_archive(drift_viterbi_bare drift_viterbi_bare_obj)
target_link_libraries(drift_viterbi_bare PUBLIC drift_viterbi_interface)
target_compile_features(drift_viterbi_bare PUBLIC c_std_11)

# The public library: the core and the stdlib shims archived together into one
# self-contained libdrift_viterbi.a. libm is carried through for the shims' math.
dv_add_archive(drift_viterbi drift_viterbi_bare_obj drift_viterbi_stdlib)
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
