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
 * Tests for dv_compare: does it recognise two streams as sharing a
 * convolutional code, including across cumulative indel drift, wide codes, and
 * short streams? Also covers the dv_compare_min_len/max_len range helpers and
 * the agreement between dv_compare and the decoder's lock probability.
 */

#include "dv_test_util.h"

#define INFO_BITS 4000
#define CODED_CAP MAX_CODED(INFO_BITS)
#define P_DEL 0.006 /* ~48 deletions over 8000 coded bits -> drift ~ -48 */

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
    check_ge("clean-same-code", dv_compare(n, k, a, la, b, lb), 0.8);

    /* Same stream skewed by a constant offset of 7 bits (<= DV_MAX_DRIFT). */
    check_ge("const-offset-7", dv_compare(n, k, a, la, a + 7, la - 7), 0.8);
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

    check_ge("cumulative-drift", dv_compare(n, k, a, la, drift, ld), 0.7);
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

    check_le("different-codes", dv_compare(n, k, a, la, drift, ld), 0.3);
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

    check_le("structured-vs-random", dv_compare(n, k, a, la, r, lr), 0.3);
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
        dv_code *enc = dv_code_create_standard(family[f][i]);
        dv_code *dec = dv_code_create_standard(family[f][j]);
        double lock =
            decoder_lock_mean(enc, dec, msg_a, info_bits, 8 * dv_code_k(dec));
        dv_code_destroy(enc);
        dv_code_destroy(dec);

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
    check_ge("short-same-code", dv_compare(n, k, a, la, b, lb), 0.7);
  }

  /* Too short to determine: a handful of coded bits -> must be undetermined. */
  {
    int n, k;
    rand_bits(msg, 6, &seed);
    size_t la = encode(DV_CODE_K7_RATE_1_2, msg, 6, a, &n, &k);
    check_undetermined("too-short", dv_compare(n, k, a, la, a, la));
  }

  free(msg);
  free(a);
  free(b);
}

/* 7. dv_compare_min_len / dv_compare_max_len report dv_compare's usable length
 *    range. min_len is a necessary floor: any sample shorter is UNDETERMINED,
 * while a comfortably longer same-code pair is determinate. max_len must exceed
 * min_len, and out-of-range (n, k) must return negative. The dv_detect_*_len
 * pair must match (same single-stream recovery). */
static void test_len_helpers(uint64_t seed) {
  printf("test_len_helpers\n");

  /* Out-of-range codes -> negative. */
  check_undetermined("compare_min_len k<2", (double)dv_compare_min_len(1, 1));
  check_undetermined("compare_max_len n<1", (double)dv_compare_max_len(0, 7));
  check_undetermined("compare_min_len window>32",
                     (double)dv_compare_min_len(5, 9));

  /* A representative in-range code. */
  dv_code *code = dv_code_create_standard(DV_CODE_K7_RATE_1_2);
  REQUIRE("code created", code != NULL);
  int n = dv_code_n(code), k = dv_code_k(code);
  dv_code_destroy(code);

  long min_len = dv_compare_min_len(n, k);
  long max_len = dv_compare_max_len(n, k);
  printf("  K7_R1_2: min_len=%ld max_len=%ld\n", min_len, max_len);
  check("min_len positive", min_len > 0);
  check("max_len > min_len", max_len > min_len);

  /* dv_detect runs the same single-stream recovery, so its length range matches
   * dv_compare's and obeys the same out-of-range contract. */
  check("detect_min_len matches compare", dv_detect_min_len(n, k) == min_len);
  check("detect_max_len matches compare", dv_detect_max_len(n, k) == max_len);
  check_undetermined("detect_min_len window>32",
                     (double)dv_detect_min_len(5, 9));
  check_undetermined("detect_max_len k<2", (double)dv_detect_max_len(1, 1));

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
  REQUIRE("encoded above min_len",
          la > (size_t)min_len && lb > (size_t)min_len);

  check_undetermined("below min_len", dv_compare(n, k, a, (size_t)min_len - 1,
                                                 b, (size_t)min_len - 1));
  check_ge("sufficient recovers", dv_compare(n, k, a, la, b, lb), 0.0);

  free(a);
  free(b);
  free(msg);
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
  return test_summary("compare");
}
