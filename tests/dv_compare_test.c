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
#include <string.h>

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

/* Encode `info_bits` random message bits with the given standard code into
 * out[] (which must hold info_bits * 5 + slack), reporting the code's n and k.
 * Returns the coded length. */
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

/* Mark each bit DV_ERASURE with probability p_erase, in place. */
static void erase_channel(uint8_t *buf, size_t len, double p_erase,
                          uint64_t *rng) {
  for (size_t i = 0; i < len; ++i) {
    if (rng_unit(rng) < p_erase) {
      buf[i] = DV_ERASURE;
    }
  }
}

#define INFO_BITS 4000
#define CODED_CAP (INFO_BITS * 5 + 64)
#define P_DEL 0.006 /* ~48 deletions over 8000 coded bits -> drift ~ -48 */

static int g_failures = 0;

static void check_ge(const char *name, int trial, double got, double want) {
  int ok = got >= want;
  printf("  [%s] trial %d: dv_compare = %.4f  (want >= %.2f)  %s\n", name,
         trial, got, want, ok ? "PASS" : "FAIL");
  if (!ok) {
    ++g_failures;
  }
}

static void check_le(const char *name, int trial, double got, double want) {
  int ok = got <= want;
  printf("  [%s] trial %d: dv_compare = %.4f  (want <= %.2f)  %s\n", name,
         trial, got, want, ok ? "PASS" : "FAIL");
  if (!ok) {
    ++g_failures;
  }
}

/* Mean decoder lock probability when code `dec_code`'s decoder reads a clean
 * stream coded with `enc_code`, over the settled second half. This is the
 * decoder-side analog of dv_compare: high when the two codes are the same, low
 * when they differ. */
static double lock_mean(dv_standard_code enc_code, dv_standard_code dec_code,
                        const uint8_t *msg, int info_bits) {
  int n, k;
  uint8_t *coded = malloc(CODED_CAP);
  size_t coded_len = encode(enc_code, msg, info_bits, coded, &n, &k);

  dv_code *dec = dv_code_create_standard(dec_code);
  assert(dec);
  dv_stream_params params = {
      .decision_depth = 8 * dv_code_k(dec),
      .max_drift = 4,
      .p_sub = 0.01,
      .p_ins = 0.01,
      .p_del = 0.01,
  };
  dv_stream_decoder *sd = dv_stream_decoder_create(dec, &params);
  assert(sd);

  int cap = (int)coded_len + 64;
  uint8_t *out = malloc((size_t)cap);
  double *lock = malloc((size_t)cap * sizeof(double));
  int got = dv_stream_decode(sd, coded, (int)coded_len, out, lock, cap);
  assert(got > 0);

  double sum = 0.0;
  int count = 0;
  for (int i = got / 2; i < got; ++i) {
    sum += lock[i];
    ++count;
  }

  dv_stream_decoder_destroy(sd);
  dv_code_destroy(dec);
  free(coded);
  free(out);
  free(lock);
  return count ? sum / count : 0.0;
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

/* 3. Different codes (same rate/K, different polynomials), one drifted: must
 * NOT be confused. Guards the offset path against manufacturing satisfaction.
 */
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

/* 5. The decoder's lock_probability and dv_compare are two routes to the same
 *    yes/no question - "do these bits belong to this code?" - so they should
 * agree. For each ordered pair within a family we decode code A's stream with
 * code B's decoder (lock) and run dv_compare on independent A and B streams,
 * then check both read "same" exactly on the diagonal (A == B) and "different"
 * off it, matching each other and the ground truth. All four families are
 * covered, including K7_R1_3 (window 24) and K5_R1_5 (window 30), which exceed
 * the WHT cap and exercise dv_compare's null-space recovery path. */
static void test_lock_matches_compare(uint64_t seed) {
  printf("test_lock_matches_compare\n");
  const dv_standard_code family[][3] = {
      {DV_CODE_K3_RATE_1_2, DV_CODE_K3_RATE_1_2_ALT1, DV_CODE_K3_RATE_1_2_ALT2},
      {DV_CODE_K7_RATE_1_2, DV_CODE_K7_RATE_1_2_ALT1, DV_CODE_K7_RATE_1_2_ALT2},
      {DV_CODE_K7_RATE_1_3, DV_CODE_K7_RATE_1_3_ALT1, DV_CODE_K7_RATE_1_3_ALT2},
      {DV_CODE_K5_RATE_1_5, DV_CODE_K5_RATE_1_5_ALT1, DV_CODE_K5_RATE_1_5_ALT2},
  };
  const char *family_name[] = {"K3_R1_2", "K7_R1_2", "K7_R1_3", "K5_R1_5"};
  const int n_families = 4;
  const int info_bits = 1500;

  /* Score > this reads as "same code". Both methods' self/sibling values sit
   * clearly either side of these boundaries (lock: self > 0.9, sibling < 0.75;
   * compare: self ~ 1.0, sibling ~ 0.25), so the classification is robust. */
  const double LOCK_SAME = 0.8;
  const double COMPARE_SAME = 0.5;

  uint8_t *msg_a = malloc((size_t)info_bits);
  uint8_t *msg_b = malloc((size_t)info_bits);
  uint8_t *coded_a = malloc(CODED_CAP);
  uint8_t *coded_b = malloc(CODED_CAP);

  for (int f = 0; f < n_families; ++f) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        uint64_t rng = seed + (uint64_t)(f * 9 + i * 3 + j) + 1;
        rand_bits(msg_a, info_bits, &rng);
        rand_bits(msg_b, info_bits, &rng);

        /* lock: decode code i's stream with code j's decoder. */
        double lock = lock_mean(family[f][i], family[f][j], msg_a, info_bits);

        /* compare: independent streams from code i and code j. */
        int n, k, n2, k2;
        size_t len_a = encode(family[f][i], msg_a, info_bits, coded_a, &n, &k);
        size_t len_b =
            encode(family[f][j], msg_b, info_bits, coded_b, &n2, &k2);
        double compare = dv_compare(n, k, coded_a, len_a, coded_b, len_b);

        int truth_same = (i == j);
        int lock_same = lock > LOCK_SAME;
        int compare_same = compare > COMPARE_SAME;
        int ok = (lock_same == truth_same) && (compare_same == truth_same) &&
                 (lock_same == compare_same);
        printf("  %s [%d->%d] lock=%.3f compare=%.3f  truth=%-4s  %s\n",
               family_name[f], i, j, lock, compare,
               truth_same ? "same" : "diff", ok ? "PASS" : "FAIL");
        if (!ok) {
          ++g_failures;
        }
      }
    }
  }
  free(msg_a);
  free(msg_b);
  free(coded_a);
  free(coded_b);
}

/* 6. Stream length: a moderately short same-code pair still recovers, while a
 *    genuinely too-short stream returns DV_UNDETERMINED (< 0) rather than a
 * spurious verdict. Confirms the lowered length floor degrades gracefully. */
static void test_short_stream(uint64_t seed) {
  printf("test_short_stream\n");
  uint8_t *msg = malloc(CODED_CAP);
  uint8_t *a = malloc(CODED_CAP);
  uint8_t *b = malloc(CODED_CAP);

  /* Short but determinable: ~150 info bits -> ~300 coded bits of K7_R1_2. */
  for (int t = 0; t < 3; ++t) {
    uint64_t rng = seed + 500 + (uint64_t)t;
    int n, k;
    rand_bits(msg, 150, &rng);
    size_t la = encode(DV_CODE_K7_RATE_1_2, msg, 150, a, &n, &k);
    rand_bits(msg, 150, &rng);
    size_t lb = encode(DV_CODE_K7_RATE_1_2, msg, 150, b, &n, &k);
    check_ge("short-same-code", t, dv_compare(n, k, a, la, b, lb), 0.7);
  }

  /* Too short to determine: a handful of coded bits -> must be undetermined. */
  {
    int n, k;
    rand_bits(msg, 6, &seed);
    size_t la = encode(DV_CODE_K7_RATE_1_2, msg, 6, a, &n, &k);
    double got = dv_compare(n, k, a, la, a, la);
    int ok = got < 0.0;
    printf("  [too-short] dv_compare = %.4f  (want < 0 / undetermined)  %s\n",
           got, ok ? "PASS" : "FAIL");
    if (!ok) {
      ++g_failures;
    }
  }

  free(msg);
  free(a);
  free(b);
}

/* 7. dv_compare_min_len / dv_compare_max_len report dv_compare's usable length
 *    range. min_len is a necessary floor: any sample shorter is UNDETERMINED,
 * while a comfortably longer same-code pair is determinate. max_len must exceed
 * min_len, and out-of-range (n, k) must return negative. */
static void test_len_helpers(uint64_t seed) {
  printf("test_len_helpers\n");
  int fail = 0;

  /* Out-of-range codes -> negative. */
  fail += !(dv_compare_min_len(1, 1) < 0); /* k < 2 */
  fail += !(dv_compare_max_len(0, 7) < 0); /* n < 1 */
  fail += !(dv_compare_min_len(5, 9) < 0); /* window 5*10 = 50 > 32 */

  /* A representative in-range code. */
  dv_code *code = dv_code_create_standard(DV_CODE_K7_RATE_1_2);
  assert(code);
  int n = dv_code_n(code), k = dv_code_k(code);
  dv_code_destroy(code);

  long min_len = dv_compare_min_len(n, k);
  long max_len = dv_compare_max_len(n, k);
  fail += !(min_len > 0);
  fail += !(max_len > min_len);
  printf("  K7_R1_2: min_len=%ld max_len=%ld\n", min_len, max_len);

  /* dv_detect runs the same single-stream recovery, so its length range matches
   * dv_compare's and obeys the same out-of-range contract. */
  fail += !(dv_detect_min_len(n, k) == min_len);
  fail += !(dv_detect_max_len(n, k) == max_len);
  fail += !(dv_detect_min_len(5, 9) < 0); /* window 5*10 = 50 > 32 */
  fail += !(dv_detect_max_len(1, 1) < 0); /* k < 2 */

  /* Below min_len dv_compare must always be UNDETERMINED (the firm guarantee);
   * a sample comfortably above the floor recovers. Distinct messages so it is a
   * genuine recovery, not a self-comparison. */
  const int info = 4 * (int)min_len; /* coded length ~8*min_len bits */
  uint8_t *a = malloc(CODED_CAP);
  uint8_t *b = malloc(CODED_CAP);
  uint8_t *msg = malloc((size_t)info);

  rand_bits(msg, info, &seed);
  size_t la = encode(DV_CODE_K7_RATE_1_2, msg, info, a, &n, &k);
  rand_bits(msg, info, &seed);
  size_t lb = encode(DV_CODE_K7_RATE_1_2, msg, info, b, &n, &k);
  assert(la > (size_t)min_len && lb > (size_t)min_len);

  double below =
      dv_compare(n, k, a, (size_t)min_len - 1, b, (size_t)min_len - 1);
  double sufficient = dv_compare(n, k, a, la, b, lb);
  int ok_below = below < 0.0;            /* too short -> undetermined */
  int ok_sufficient = sufficient >= 0.0; /* enough data -> determinate */
  printf("  below min_len: %.4f (%s);  sufficient: %.4f (%s)\n", below,
         ok_below ? "undetermined" : "DETERMINATE", sufficient,
         ok_sufficient ? "determinate" : "UNDETERMINED");
  fail += !ok_below + !ok_sufficient;

  if (fail) {
    printf("  %d length-helper check(s) FAILED\n", fail);
    g_failures += fail;
  } else {
    printf("  all length-helper checks passed\n");
  }
  free(a);
  free(b);
  free(msg);
}

/* 8. dv_detect: does a single buffer carry any code at (n, k)? Clean coded data
 *    detects high; random data low; and detection survives indels and erasures,
 * like dv_compare. Too-short, out-of-range, or null inputs are undetermined. */
static void test_detect(uint64_t seed) {
  printf("test_detect\n");
  int fail = 0;
  const dv_standard_code codes[] = {DV_CODE_K3_RATE_1_2, DV_CODE_K7_RATE_1_2,
                                    DV_CODE_K7_RATE_1_3, DV_CODE_K5_RATE_1_5};
  const char *names[] = {"K3_R1_2", "K7_R1_2", "K7_R1_3", "K5_R1_5"};
  const int info_bits = 1500;

  uint8_t *msg = malloc((size_t)info_bits);
  uint8_t *coded = malloc(CODED_CAP);
  uint8_t *chan = malloc(CODED_CAP);

  for (int c = 0; c < 4; ++c) {
    uint64_t rng = seed + 700 + (uint64_t)c;
    int n, k;
    rand_bits(msg, info_bits, &rng);
    size_t clen = encode(codes[c], msg, info_bits, coded, &n, &k);

    double clean = dv_detect(n, k, coded, clen);

    size_t dlen = delete_channel(coded, clen, 0.01, &rng, chan);
    double indel = dv_detect(n, k, chan, dlen);

    memcpy(chan, coded, clen);
    erase_channel(chan, clen, 0.04, &rng);
    double erased = dv_detect(n, k, chan, clen);

    int ok = clean > 0.8 && indel > 0.7 && erased > 0.7;
    printf("  %-7s clean=%.3f indel=%.3f erased=%.3f  %s\n", names[c], clean,
           indel, erased, ok ? "PASS" : "FAIL");
    fail += !ok;
  }

  /* Random (non-coded) buffer -> low. */
  {
    uint64_t rng = seed + 800;
    rand_bits(coded, 3000, &rng);
    double rnd = dv_detect(2, 7, coded, 3000); /* K7_R1_2 parameters */
    int ok = rnd < 0.3;
    printf("  random:  %.3f  %s\n", rnd, ok ? "PASS" : "FAIL");
    fail += !ok;
  }

  /* Too short, out-of-range, and null -> undetermined (negative). */
  fail += !(dv_detect(2, 7, coded, 8) < 0);    /* below one window */
  fail += !(dv_detect(5, 9, coded, 3000) < 0); /* window 5*10 = 50 > 32 */
  fail += !(dv_detect(2, 7, NULL, 3000) < 0);  /* null buffer */

  if (fail) {
    printf("  %d detect check(s) FAILED\n", fail);
    g_failures += fail;
  } else {
    printf("  all detect checks passed\n");
  }
  free(msg);
  free(coded);
  free(chan);
}

int main(void) {
  const uint64_t seed = 0xD1F7C0DEULL;
  test_clean_and_constant_offset(seed);
  test_cumulative_drift_same_code(seed);
  test_different_codes_negative(seed);
  test_structured_vs_random(seed);
  test_lock_matches_compare(seed);
  test_short_stream(seed);
  test_len_helpers(seed);
  test_detect(seed);

  if (g_failures) {
    printf("\n%d check(s) FAILED\n", g_failures);
    return 1;
  }
  printf("\nall compare checks passed\n");
  return 0;
}
