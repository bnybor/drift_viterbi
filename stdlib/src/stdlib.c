#include <drift_viterbi/stdlib.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

void *dv_malloc(size_t size) { return malloc(size); }
void *dv_calloc(size_t count, size_t size) { return calloc(count, size); }
void *dv_realloc(void *ptr, size_t size) { return realloc(ptr, size); }
void dv_free(void *ptr) { free(ptr); }

void *dv_memcpy(void *dest, const void *src, size_t n) {
  return memcpy(dest, src, n);
}
void *dv_memmove(void *dest, const void *src, size_t n) {
  return memmove(dest, src, n);
}
void *dv_memset(void *dest, int value, size_t n) {
  return memset(dest, value, n);
}

int dv_abs(int value) { return abs(value); }
double dv_log(double x) { return log(x); }
double dv_exp(double x) { return exp(x); }
