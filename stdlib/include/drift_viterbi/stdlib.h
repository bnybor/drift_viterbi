#ifndef STDLIB_INCLUDE_DRIFT_VITERBI_STDLIB_H_
#define STDLIB_INCLUDE_DRIFT_VITERBI_STDLIB_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The freestanding core (encode.c, decode.c, compare.c) is built -nostdlib
 * -ffreestanding -fno-builtin and must not pull in the C standard library
 * directly. It reaches the few libc facilities it needs through these dv_*
 * proxies instead, so this translation unit (stdlib.c) is the single, explicit
 * boundary where libc is touched. Each proxy mirrors its standard counterpart's
 * signature, so a declaration is always in scope at the call site (an implicitly
 * declared allocator would be assumed to return int, truncating 64-bit pointers).
 */

void *dv_malloc(size_t size);
void *dv_calloc(size_t count, size_t size);
void *dv_realloc(void *ptr, size_t size);
void dv_free(void *ptr);

void *dv_memcpy(void *dest, const void *src, size_t n);
void *dv_memmove(void *dest, const void *src, size_t n);
void *dv_memset(void *dest, int value, size_t n);

int dv_abs(int value);
double dv_log(double x);
double dv_exp(double x);

#ifdef __cplusplus
}
#endif

#endif /* STDLIB_INCLUDE_DRIFT_VITERBI_STDLIB_H_ */
