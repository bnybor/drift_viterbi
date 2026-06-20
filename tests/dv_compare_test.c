/* clang-format off */
/*
 * MIT License
 *
 * Copyright (c) 2026 Robyn Kirkman
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
/* clang-format on */

/*
 * Tests for dv_compare (compare.c): does it recognise two streams as
 * sharing a convolutional code, including when one has accumulated mid-stream
 * insertion/deletion drift well past the constant-offset tolerance?
 */

#include <drift_viterbi/drift_viterbi.h>

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* -- deterministic PRNG (splitmix64) --------------------------------------- */

static uint64_t rng_next(uint64_t *state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
static double rng_unit(uint64_t *state) {
  return (double)(rng_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

static void rand_bits(uint8_t *bits, int n, uint64_t *rng) {
  for (int i = 0; i < n; ++i) {
    bits[i] = (uint8_t)(rng_next(rng) & 1u);
  }
}

/* -- helpers --------------------------------------------------------------- */

/* Encode `info_bits` random message bits with the given standard code into out[]
 * (which must hold info_bits * 5 + slack), reporting the code's n and k. Returns
 * the coded length. */
static size_t encode(dv_standard_code which, const uint8_t *msg, int info_bits,
                     uint8_t *out, int *n, int *k) {
  dv_code *c = dv_code_create_standard(which);
  assert(c);
  *n = dv_code_n(c);
  *k = dv_code_k(c);
  int state = 0;
  int len = dv_code_encode(c, msg, info_bits, &state, out);
  len += dv_code_encode_flush(c, &state, out + len);
  dv_code_destroy(c);
  return (size_t)len;
}

/* Drop each coded bit with probability p_del; monotonic drift grows with
 * position. Returns the received length. */
static size_t delete_channel(const uint8_t *in, size_t len, double p_del,
                             uint64_t *rng, uint8_t *out) {
  size_t o = 0;
  for (size_t i = 0; i < len; ++i) {
    if (rng_unit(rng) < p_del) {
      continue;
    }
    out[o++] = in[i];
  }
  return o;
}

#define INFO_BITS 4000
#define CODED_CAP (INFO_BITS * 5 + 64)
#define P_DEL 0.006 /* ~48 deletions over 8000 coded bits -> drift ~ -48 */

static int g_failures = 0;

static void check_ge(const char *name, int trial, double got, double want) {
  int ok = got >= want;
  printf("  [%s] trial %d: dv_compare = %.4f  (want >= %.2f)  %s\n", name, trial,
         got, want, ok ? "PASS" : "FAIL");
  if (!ok) {
    ++g_failures;
  }
}

static void check_le(const char *name, int trial, double got, double want) {
  int ok = got <= want;
  printf("  [%s] trial %d: dv_compare = %.4f  (want <= %.2f)  %s\n", name, trial,
         got, want, ok ? "PASS" : "FAIL");
  if (!ok) {
    ++g_failures;
  }
}

/* -- tests ----------------------------------------------------------------- */

/* 1. Same code, no indels: clean agreement and a constant within-range skew. */
static void test_clean_and_constant_offset(uint64_t seed) {
  printf("test_clean_and_constant_offset\n");
  uint8_t *msg = malloc(INFO_BITS);
  uint8_t *a = malloc(CODED_CAP);
  uint8_t *b = malloc(CODED_CAP);
  for (int t = 0; t < 3; ++t) {
    uint64_t rng = seed + (uint64_t)t;
    int n, k;
    rand_bits(msg, INFO_BITS, &rng);
    size_t la = encode(DV_CODE_K7_RATE_1_2, msg, INFO_BITS, a, &n, &k);
    rand_bits(msg, INFO_BITS, &rng);
    size_t lb = encode(DV_CODE_K7_RATE_1_2, msg, INFO_BITS, b, &n, &k);

    /* Two independent clean encodings of the same code. */
    check_ge("clean-same-code", t, dv_compare(n, k, a, la, b, lb), 0.8);

    /* Same stream skewed by a constant offset of 7 bits (<= DV_MAX_DRIFT). */
    check_ge("const-offset-7", t, dv_compare(n, k, a, la, a + 7, la - 7), 0.8);
  }
  free(msg);
  free(a);
  free(b);
}

/* 2. Same code, one stream through a deletion channel that drifts well past the
 *    16-bit constant-offset limit. This is the headline capability. */
static void test_cumulative_drift_same_code(uint64_t seed) {
  printf("test_cumulative_drift_same_code\n");
  uint8_t *msg = malloc(INFO_BITS);
  uint8_t *a = malloc(CODED_CAP);
  uint8_t *c = malloc(CODED_CAP);
  uint8_t *drift = malloc(CODED_CAP);
  for (int t = 0; t < 3; ++t) {
    uint64_t rng = seed + 100 + (uint64_t)t;
    int n, k;
    rand_bits(msg, INFO_BITS, &rng);
    size_t la = encode(DV_CODE_K7_RATE_1_2, msg, INFO_BITS, a, &n, &k);
    rand_bits(msg, INFO_BITS, &rng);
    size_t lc = encode(DV_CODE_K7_RATE_1_2, msg, INFO_BITS, c, &n, &k);
    size_t ld = delete_channel(c, lc, P_DEL, &rng, drift);

    check_ge("cumulative-drift", t, dv_compare(n, k, a, la, drift, ld), 0.7);
  }
  free(msg);
  free(a);
  free(c);
  free(drift);
}

/* 3. Different codes (same rate/K, different polynomials), one drifted: must NOT
 *    be confused. Guards the offset path against manufacturing satisfaction. */
static void test_different_codes_negative(uint64_t seed) {
  printf("test_different_codes_negative\n");
  uint8_t *msg = malloc(INFO_BITS);
  uint8_t *a = malloc(CODED_CAP);
  uint8_t *d = malloc(CODED_CAP);
  uint8_t *drift = malloc(CODED_CAP);
  for (int t = 0; t < 3; ++t) {
    uint64_t rng = seed + 200 + (uint64_t)t;
    int n, k, n2, k2;
    rand_bits(msg, INFO_BITS, &rng);
    size_t la = encode(DV_CODE_K7_RATE_1_2, msg, INFO_BITS, a, &n, &k);
    rand_bits(msg, INFO_BITS, &rng);
    size_t ldc = encode(DV_CODE_K7_RATE_1_2_ALT1, msg, INFO_BITS, d, &n2, &k2);
    size_t ld = delete_channel(d, ldc, P_DEL, &rng, drift);

    check_le("different-codes", t, dv_compare(n, k, a, la, drift, ld), 0.3);
  }
  free(msg);
  free(a);
  free(d);
  free(drift);
}

/* 4. Structured stream vs. unstructured random: must read as different. */
static void test_structured_vs_random(uint64_t seed) {
  printf("test_structured_vs_random\n");
  uint8_t *msg = malloc(INFO_BITS);
  uint8_t *a = malloc(CODED_CAP);
  uint8_t *r = malloc(CODED_CAP);
  for (int t = 0; t < 3; ++t) {
    uint64_t rng = seed + 300 + (uint64_t)t;
    int n, k;
    rand_bits(msg, INFO_BITS, &rng);
    size_t la = encode(DV_CODE_K7_RATE_1_2, msg, INFO_BITS, a, &n, &k);
    size_t lr = la;
    rand_bits(r, (int)lr, &rng);

    check_le("structured-vs-random", t, dv_compare(n, k, a, la, r, lr), 0.3);
  }
  free(msg);
  free(a);
  free(r);
}

int main(void) {
  const uint64_t seed = 0xD1F7C0DEULL;
  test_clean_and_constant_offset(seed);
  test_cumulative_drift_same_code(seed);
  test_different_codes_negative(seed);
  test_structured_vs_random(seed);

  if (g_failures) {
    printf("\n%d check(s) FAILED\n", g_failures);
    return 1;
  }
  printf("\nall compare checks passed\n");
  return 0;
}
