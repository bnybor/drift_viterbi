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
 * Shared test utilities for the drift_viterbi suite: a small soft-assertion
 * framework plus the PRNG, encode, channel, and decoder helpers the test files
 * have in common. Header-only; every helper is `static inline` so a test file
 * that does not use one draws no -Wunused warning.
 */

#ifndef DV_TEST_UTIL_H
#define DV_TEST_UTIL_H

#include <drift_viterbi/drift_viterbi.h>

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -- check framework ------------------------------------------------------- */

/* Failures counted per test executable (each test file is its own program).
 *
 * The check helpers below run inside OpenMP-parallelised test loops, so every
 * one updates `g_failures` and writes its PASS/FAIL line under the same named
 * critical section: the counter stays race-free and lines never tear. Without
 * OpenMP the `#pragma omp critical` is an ignored unknown pragma, so the same
 * source compiles and behaves identically single-threaded. */
static int g_failures = 0;

/* Record a boolean outcome; print PASS/FAIL and return the condition so callers
 * can branch (see REQUIRE). */
static inline int check(const char *name, int cond) {
#pragma omp critical(dv_check)
  {
    printf("  [%s] %s\n", name, cond ? "PASS" : "FAIL");
    if (!cond) {
      ++g_failures;
    }
  }
  return cond;
}

static inline void check_ge(const char *name, double got, double want) {
  int ok = got >= want;
#pragma omp critical(dv_check)
  {
    printf("  [%s] %.4f (want >= %.4f) %s\n", name, got, want,
           ok ? "PASS" : "FAIL");
    if (!ok) {
      ++g_failures;
    }
  }
}

static inline void check_gt(const char *name, double got, double want) {
  int ok = got > want;
#pragma omp critical(dv_check)
  {
    printf("  [%s] %.4f (want > %.4f) %s\n", name, got, want,
           ok ? "PASS" : "FAIL");
    if (!ok) {
      ++g_failures;
    }
  }
}

static inline void check_le(const char *name, double got, double want) {
  int ok = got <= want;
#pragma omp critical(dv_check)
  {
    printf("  [%s] %.4f (want <= %.4f) %s\n", name, got, want,
           ok ? "PASS" : "FAIL");
    if (!ok) {
      ++g_failures;
    }
  }
}

static inline void check_lt(const char *name, double got, double want) {
  int ok = got < want;
#pragma omp critical(dv_check)
  {
    printf("  [%s] %.4f (want < %.4f) %s\n", name, got, want,
           ok ? "PASS" : "FAIL");
    if (!ok) {
      ++g_failures;
    }
  }
}

/* A negative value is the "result could not be determined" signal. */
static inline void check_undetermined(const char *name, double got) {
  int ok = got < 0.0;
#pragma omp critical(dv_check)
  {
    printf("  [%s] %.4f (want < 0, undetermined) %s\n", name, got,
           ok ? "PASS" : "FAIL");
    if (!ok) {
      ++g_failures;
    }
  }
}

/* Fatal precondition: on failure record it and bail out of the current test
 * (which must return void) rather than press on and dereference a NULL. */
#define REQUIRE(name, cond)       \
  do {                            \
    if (!check((name), (cond))) { \
      return;                     \
    }                             \
  } while (0)

/* Print the suite summary and yield a process exit code for main to return. */
static inline int test_summary(const char *suite) {
  if (g_failures) {
    printf("\n%s: %d check(s) FAILED\n", suite, g_failures);
    return 1;
  }
  printf("\nall %s checks passed\n", suite);
  return 0;
}

/* -- deterministic PRNG (splitmix64) --------------------------------------- */

static inline uint64_t rng_next(uint64_t *state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

static inline double rng_unit(uint64_t *state) {
  return (double)(rng_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

static inline void rand_bits(uint8_t *bits, int n, uint64_t *rng) {
  for (int i = 0; i < n; ++i) {
    bits[i] = (uint8_t)(rng_next(rng) & 1u);
  }
}

/* -- encode / channels ----------------------------------------------------- */

/* Capacity (bits) that holds the encode of `info` info bits for any standard
 * code (max rate is 1/5) plus flush slack. */
#define MAX_CODED(info) ((info) * 5 + 64)

/* Encode `info_bits` message bits with the given standard code into out[]
 * (which must hold MAX_CODED(info_bits)), reporting the code's n and k. Returns
 * the coded length. */
static inline size_t encode(dv_standard_code which, const uint8_t *msg,
                            int info_bits, uint8_t *out, int *n, int *k) {
  dv_code *code = dv_code_create_standard(which);
  assert(code);
  *n = dv_code_n(code);
  *k = dv_code_k(code);
  int state = 0;
  int len = dv_code_encode(code, msg, info_bits, &state, out);
  len += dv_code_encode_flush(code, &state, out + len);
  dv_code_destroy(code);
  return (size_t)len;
}

/* Drop each coded bit with probability p_del; cumulative drift grows with
 * position. Returns the received length. */
static inline size_t delete_channel(const uint8_t *in, size_t len, double p_del,
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

/* Mark each bit DV_ERASURE with probability p_erase, in place. */
static inline void erase_channel(uint8_t *buf, size_t len, double p_erase,
                                 uint64_t *rng) {
  for (size_t i = 0; i < len; ++i) {
    if (rng_unit(rng) < p_erase) {
      buf[i] = DV_ERASURE;
    }
  }
}

/* Prepend `plen` random bits before `body` (blen bits) into out[], modelling a
 * capture that begins on garbage / a partial codeword before the coded stream
 * starts. Returns the combined length. out[] must hold plen + blen bits. */
static inline size_t prepend_prefix(size_t plen, const uint8_t *body,
                                    size_t blen, uint64_t *rng, uint8_t *out) {
  for (size_t i = 0; i < plen; ++i) {
    out[i] = (uint8_t)(rng_next(rng) & 1u);
  }
  memcpy(out + plen, body, blen);
  return plen + blen;
}

/* -- decoder helpers ------------------------------------------------------- */

/* Build a decoder from positional settings (keeps tests concise). */
static inline dv_stream_decoder *make_decoder(const dv_code *code, int depth,
                                              int drift, double p_sub,
                                              double p_ins, double p_del,
                                              double p_erase) {
  dv_stream_params params = {
      .decision_depth = depth,
      .max_drift = drift,
      .p_sub = p_sub,
      .p_ins = p_ins,
      .p_del = p_del,
      .p_erase = p_erase,
  };
  return dv_stream_decoder_create(code, &params);
}

/* Push a whole received buffer through the streaming decoder in small chunks,
 * then drain. Returns the number of decoded bits collected. */
static inline int stream_decode_all(dv_stream_decoder *sd, const uint8_t *rx,
                                    int rl, uint8_t *out, int cap) {
  int got = 0;
  for (int pos = 0; pos < rl;) {
    int chunk = (rl - pos < 41) ? (rl - pos) : 41;
    int w = dv_stream_decode(sd, rx + pos, chunk, out + got, NULL, cap - got);
    assert(w >= 0);
    got += w;
    pos += chunk;
  }
  for (;;) {
    int w = dv_stream_decode_flush(sd, out + got, cap - got);
    assert(w >= 0);
    if (w == 0) break;
    got += w;
  }
  return got;
}

/* Encode `msg` (info_bits bits) with `enc`, decode it with a fresh decoder for
 * `dec` at the given decision depth, and return the mean lock probability over
 * the settled second half. When enc != dec this measures whether one code's
 * stream is mistaken for the other's. */
static inline double decoder_lock_mean(const dv_code *enc, const dv_code *dec,
                                       const uint8_t *msg, int info_bits,
                                       int depth) {
  int clen = info_bits * dv_code_n(enc);
  uint8_t *coded = malloc((size_t)clen);
  int st = 0;
  dv_code_encode(enc, msg, info_bits, &st, coded);

  dv_stream_decoder *sd = make_decoder(dec, depth, 4, 0.01, 0.01, 0.01, 0.0);
  assert(sd != NULL);
  int cap = info_bits + 64;
  uint8_t *out = malloc((size_t)cap);
  double *lock = malloc((size_t)cap * sizeof(double));
  int got = dv_stream_decode(sd, coded, clen, out, lock, cap);
  assert(got > 0);

  double sum = 0.0;
  int count = 0;
  for (int i = got / 2; i < got; ++i) {
    sum += lock[i];
    ++count;
  }

  dv_stream_decoder_destroy(sd);
  free(coded);
  free(out);
  free(lock);
  return count ? sum / count : 0.0;
}

#endif /* DV_TEST_UTIL_H */
