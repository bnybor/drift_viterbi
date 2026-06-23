add_library(drift_viterbi_stdlib_interface INTERFACE)
target_include_directories(drift_viterbi_stdlib_interface INTERFACE include)

# OBJECT library so its compiled object is archived directly into the public
# drift_viterbi library.
add_library(drift_viterbi_stdlib OBJECT src/stdlib.c)
target_link_libraries(drift_viterbi_stdlib PUBLIC m drift_viterbi_stdlib_interface)
