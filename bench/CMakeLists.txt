add_executable(dv_metrics dv_metrics.c)
target_link_libraries(dv_metrics PRIVATE drift_viterbi::drift_viterbi)

# Selects the standard codes' generator polynomials (see src/encode.c).
add_executable(dv_codesearch dv_codesearch.c)
target_link_libraries(dv_codesearch PRIVATE drift_viterbi::drift_viterbi)

# Fan the independent work out across cores when OpenMP is available. Without it
# the code still builds and runs single-threaded (the pragmas are simply
# ignored).
find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(dv_metrics PRIVATE OpenMP::OpenMP_C)
    target_link_libraries(dv_codesearch PRIVATE OpenMP::OpenMP_C)
endif()
