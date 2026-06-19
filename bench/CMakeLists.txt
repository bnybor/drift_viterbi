add_executable(dv_metrics dv_metrics.c)
target_link_libraries(dv_metrics PRIVATE drift_viterbi::drift_viterbi)

# Fan the independent sweep points out across cores when OpenMP is available.
# Without it the code still builds and runs single-threaded (the pragmas are
# simply ignored).
find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(dv_metrics PRIVATE OpenMP::OpenMP_C)
endif()
