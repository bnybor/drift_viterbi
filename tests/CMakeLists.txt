# The heavier suites (compare/decoder) drive embarrassingly parallel
# Monte-Carlo loops; OpenMP fans those across cores. It is optional - without it
# the `#pragma omp` lines are ignored and the tests simply run single-threaded.
find_package(OpenMP COMPONENTS C QUIET)

foreach(t decoder_test compare_test detect_test)
    add_executable(${t} ${t}.c)
    target_link_libraries(${t} PRIVATE drift_viterbi::drift_viterbi)
    if(OpenMP_C_FOUND)
        target_link_libraries(${t} PRIVATE OpenMP::OpenMP_C)
    endif()
    add_test(NAME ${t} COMMAND ${t})
endforeach()
