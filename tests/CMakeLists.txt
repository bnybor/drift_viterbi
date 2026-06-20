add_executable(test_drift_viterbi test_drift_viterbi.c)
target_link_libraries(test_drift_viterbi PRIVATE drift_viterbi::drift_viterbi)

add_test(NAME test_drift_viterbi COMMAND test_drift_viterbi)

add_executable(dv_compare_test dv_compare_test.c)
target_link_libraries(dv_compare_test PRIVATE drift_viterbi::drift_viterbi)

add_test(NAME dv_compare_test COMMAND dv_compare_test)
