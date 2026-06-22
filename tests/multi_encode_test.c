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
 * Tests for the multi-encoder: holding a same-rate, same-K family of codes over
 * one shared encoder state and emitting whichever code is selected by index.
 * Covers equivalence with the single encoder, that the shared state carries
 * across chunked calls and across a mid-stream code switch, a round-trip through
 * the multi-decoder, and argument/edge handling. Encoding is deterministic, so
 * these run serially (no Monte-Carlo).
 */

#include "dv_test_util.h"

/* Three mutually distinguishable rate-1/3, K=7 codes: same n and constraint
 * length, different generators - the family the multi-encoder expects. */
static const dv_standard_code FAMILY[] = {
    DV_CODE_K7_RATE_1_3,
    DV_CODE_K7_RATE_1_3_ALT1,
    DV_CODE_K7_RATE_1_3_ALT2,
};
#define N_FAM ((int)(sizeof(FAMILY) / sizeof(FAMILY[0])))
#define N_INFO 256
#define DEPTH 48

/* Build a multi-encoder over the whole family, handing back the codes (which
 * must outlive the encoder) via `codes`. Fresh codes => the shared state is 0. */
static dv_multi_encoder *build_multi_encoder(dv_code *codes[N_FAM]) {
  const dv_code *cp[N_FAM];
  for (int j = 0; j < N_FAM; ++j) {
    codes[j] = dv_code_create_standard(FAMILY[j]);
    assert(codes[j]);
    cp[j] = codes[j];
  }
  dv_multi_encode_params params = {.codes = cp, .codes_len = N_FAM};
  return dv_multi_encode_create(&params);
}

static void destroy_codes(dv_code *codes[N_FAM]) {
  for (int j = 0; j < N_FAM; ++j) {
    dv_code_destroy(codes[j]);
  }
}

/* Each code, encoded through the multi-encoder, is byte-identical to the single
 * encoder over the same code from state 0. */
static void test_single_index_equivalence(uint64_t seed) {
  printf("test_single_index_equivalence\n");
  uint64_t rng = seed;
  uint8_t msg[N_INFO];
  rand_bits(msg, N_INFO, &rng);

  for (int j = 0; j < N_FAM; ++j) {
    dv_code *codes[N_FAM];
    dv_multi_encoder *e = build_multi_encoder(codes);
    REQUIRE("equiv: encoder created", e != NULL);

    uint8_t got[MAX_CODED(N_INFO)];
    int w = dv_multi_encode(e, j, msg, N_INFO, got, (int)sizeof got);
    int wf = dv_multi_encode_flush(e, j, got + w, (int)sizeof got - w);

    uint8_t ref[MAX_CODED(N_INFO)];
    int n, k;
    size_t rl = encode(FAMILY[j], msg, N_INFO, ref, &n, &k);

    check("equiv: length matches single encoder", (size_t)(w + wf) == rl);
    check("equiv: bytes match single encoder",
          w + wf > 0 && memcmp(got, ref, (size_t)(w + wf)) == 0);

    dv_multi_encode_destroy(e);
    destroy_codes(codes);
  }
}

/* The shared state carries across calls: encoding a message in chunks gives the
 * same stream as encoding it in one shot. */
static void test_chunked_matches_oneshot(uint64_t seed) {
  printf("test_chunked_matches_oneshot\n");
  uint64_t rng = seed;
  uint8_t msg[N_INFO];
  rand_bits(msg, N_INFO, &rng);
  const int idx = 1;

  dv_code *codes_a[N_FAM];
  dv_multi_encoder *one = build_multi_encoder(codes_a);
  REQUIRE("chunk: one-shot encoder", one != NULL);
  uint8_t a[MAX_CODED(N_INFO)];
  int la = dv_multi_encode(one, idx, msg, N_INFO, a, (int)sizeof a);
  la += dv_multi_encode_flush(one, idx, a + la, (int)sizeof a - la);

  dv_code *codes_b[N_FAM];
  dv_multi_encoder *many = build_multi_encoder(codes_b);
  REQUIRE("chunk: chunked encoder", many != NULL);
  uint8_t b[MAX_CODED(N_INFO)];
  int lb = 0;
  for (int pos = 0; pos < N_INFO;) {
    int chunk = (N_INFO - pos < 37) ? (N_INFO - pos) : 37;
    lb += dv_multi_encode(many, idx, msg + pos, chunk, b + lb, (int)sizeof b - lb);
    pos += chunk;
  }
  lb += dv_multi_encode_flush(many, idx, b + lb, (int)sizeof b - lb);

  check("chunk: same length", la == lb);
  check("chunk: same bytes", la > 0 && memcmp(a, b, (size_t)la) == 0);

  dv_multi_encode_destroy(one);
  dv_multi_encode_destroy(many);
  destroy_codes(codes_a);
  destroy_codes(codes_b);
}

/* Switching the selected code mid-stream is coherent because the state is shared
 * and code-independent for a same-K family: the prefix equals code A's encoding
 * of the first half, and the suffix equals code B continued from the very same
 * state (reached by feeding code B the same first-half bits). */
static void test_shared_state_code_switch(uint64_t seed) {
  printf("test_shared_state_code_switch\n");
  uint64_t rng = seed;
  uint8_t msg[N_INFO];
  rand_bits(msg, N_INFO, &rng);
  const int a_idx = 0, b_idx = 1, half = N_INFO / 2;

  dv_code *codes[N_FAM];
  dv_multi_encoder *e = build_multi_encoder(codes);
  REQUIRE("switch: encoder created", e != NULL);
  const int n = dv_code_n(codes[0]);

  uint8_t got[MAX_CODED(N_INFO)];
  int w = dv_multi_encode(e, a_idx, msg, half, got, (int)sizeof got);
  int w2 =
      dv_multi_encode(e, b_idx, msg + half, N_INFO - half, got + w,
                      (int)sizeof got - w);
  int wf = dv_multi_encode_flush(e, b_idx, got + w + w2, (int)sizeof got - w - w2);

  /* Prefix: code A's encoding of the first half from state 0. */
  uint8_t ref_a[MAX_CODED(N_INFO)];
  int sa = 0;
  int la = dv_code_encode(codes[a_idx], msg, half, &sa, ref_a);
  check("switch: prefix is code A", w == la &&
                                        memcmp(got, ref_a, (size_t)la) == 0);

  /* Suffix: advance code B's state over the same first-half bits (output
   * discarded), then its encoding of the second half plus flush. */
  uint8_t scratch[MAX_CODED(N_INFO)];
  uint8_t ref_b[MAX_CODED(N_INFO)];
  int sb = 0;
  dv_code_encode(codes[b_idx], msg, half, &sb, scratch);
  int lb = dv_code_encode(codes[b_idx], msg + half, N_INFO - half, &sb, ref_b);
  lb += dv_code_encode_flush(codes[b_idx], &sb, ref_b + lb);
  check("switch: suffix is code B from shared state",
        w2 + wf == lb && memcmp(got + w, ref_b, (size_t)lb) == 0);
  check("switch: prefix length is half*n", w == half * n);

  dv_multi_encode_destroy(e);
  destroy_codes(codes);
}

/* Round-trip: a stream produced with code `true_idx` decodes cleanly through a
 * multi-decoder over the same family, attributed to that code. */
static void test_round_trip(uint64_t seed, int true_idx) {
  printf("test_round_trip (idx %d)\n", true_idx);
  uint64_t rng = seed;
  uint8_t msg[N_INFO];
  rand_bits(msg, N_INFO, &rng);

  dv_code *codes[N_FAM];
  dv_multi_encoder *e = build_multi_encoder(codes);
  REQUIRE("rt: encoder created", e != NULL);

  uint8_t coded[MAX_CODED(N_INFO)];
  int w = dv_multi_encode(e, true_idx, msg, N_INFO, coded, (int)sizeof coded);
  w += dv_multi_encode_flush(e, true_idx, coded + w, (int)sizeof coded - w);
  REQUIRE("rt: encoded some bits", w > 0);

  const dv_code *cp[N_FAM];
  for (int j = 0; j < N_FAM; ++j) {
    cp[j] = codes[j];
  }
  dv_multi_params dp = {.codes = cp,
                        .codes_len = N_FAM,
                        .stream = {.decision_depth = DEPTH, .p_sub = 0.02}};
  dv_multi_decoder *md = dv_multi_create(&dp);
  REQUIRE("rt: decoder created", md != NULL);

  const int cap = N_INFO + 64;
  uint8_t out[N_INFO + 64];
  int locked[N_INFO + 64];
  for (int i = 0; i < cap; ++i) {
    locked[i] = -1;
  }
  int got = 0;
  for (int pos = 0; pos < w;) {
    int chunk = (w - pos < 41) ? (w - pos) : 41;
    int x = dv_multi_decode(md, coded + pos, chunk, out + got, locked + got,
                            cap - got);
    assert(x >= 0);
    got += x;
    pos += chunk;
  }
  for (;;) {
    int x = dv_multi_decode_flush(md, out + got, cap - got);
    assert(x >= 0);
    if (x == 0) break;
    got += x;
  }

  int errors = 0, wrong = 0, tail_set = 0;
  for (int i = 0; i < got && i < N_INFO; ++i) {
    if (out[i] == DV_ERASURE) continue;
    if (out[i] != msg[i]) ++errors;
    if (locked[i] >= 0 && locked[i] != true_idx) ++wrong;
  }
  /* The flush appended K-1 zero input bits, so the decoder yields N_INFO + (K-1)
   * bits; the trailing ones decode back to zero. */
  for (int i = N_INFO; i < got; ++i) {
    if (out[i] != DV_ERASURE && out[i] != DV_FALSE) ++tail_set;
  }
  check("rt: message bits plus flush tail decoded",
        got == N_INFO + (dv_code_k(codes[0]) - 1));
  check("rt: exact recovery", errors == 0);
  check("rt: flush tail is zero", tail_set == 0);
  check("rt: attributed to true code", wrong == 0);

  dv_multi_destroy(md);
  dv_multi_encode_destroy(e);
  destroy_codes(codes);
}

static void test_args(void) {
  printf("test_args\n");
  uint8_t msg[8] = {0};
  uint8_t buf[MAX_CODED(N_INFO)];

  dv_code *codes[N_FAM];
  dv_multi_encoder *e = build_multi_encoder(codes);
  REQUIRE("args: encoder created", e != NULL);

  check("args: NULL handle", dv_multi_encode(NULL, 0, msg, 1, buf,
                                             (int)sizeof buf) == DV_ERR_ARG);
  check("args: idx < 0",
        dv_multi_encode(e, -1, msg, 1, buf, (int)sizeof buf) == DV_ERR_ARG);
  check("args: idx >= n",
        dv_multi_encode(e, N_FAM, msg, 1, buf, (int)sizeof buf) == DV_ERR_ARG);
  check("args: negative n_bits",
        dv_multi_encode(e, 0, msg, -1, buf, (int)sizeof buf) == DV_ERR_ARG);
  /* 4 bits at rate 1/3 need 12 output bits; max_out 1 is too small. */
  check("args: max_out too small (encode)",
        dv_multi_encode(e, 0, msg, 4, buf, 1) == DV_ERR_ARG);
  /* flush writes (7-1)*3 = 18 bits; max_out 1 is too small. */
  check("args: max_out too small (flush)",
        dv_multi_encode_flush(e, 0, buf, 1) == DV_ERR_ARG);
  check("args: NULL handle (flush)",
        dv_multi_encode_flush(NULL, 0, buf, (int)sizeof buf) == DV_ERR_ARG);

  dv_multi_encode_destroy(e);
  destroy_codes(codes);

  /* Empty set creates fine but cannot encode (no valid index). */
  dv_multi_encode_params ep = {.codes = NULL, .codes_len = 0};
  dv_multi_encoder *empty = dv_multi_encode_create(&ep);
  check("args: empty set creates", empty != NULL);
  check("args: empty set encode errors",
        dv_multi_encode(empty, 0, msg, 1, buf, (int)sizeof buf) == DV_ERR_ARG);
  dv_multi_encode_destroy(empty);

  check("args: NULL params", dv_multi_encode_create(NULL) == NULL);

  /* A NULL code entry is rejected at create. */
  dv_code *c0 = dv_code_create_standard(FAMILY[0]);
  assert(c0);
  const dv_code *with_null[2] = {c0, NULL};
  dv_multi_encode_params np = {.codes = with_null, .codes_len = 2};
  check("args: NULL code entry rejected", dv_multi_encode_create(&np) == NULL);

  /* A mismatched rate/constraint length is rejected at create. */
  dv_code *k3 = dv_code_create_standard(DV_CODE_K3_RATE_1_2);
  assert(k3);
  const dv_code *mixed[2] = {c0, k3};
  dv_multi_encode_params mp = {.codes = mixed, .codes_len = 2};
  check("args: mismatched n/k rejected", dv_multi_encode_create(&mp) == NULL);

  dv_code_destroy(c0);
  dv_code_destroy(k3);
}

int main(void) {
  const uint64_t seed = 0xC0FFEEULL;
  test_single_index_equivalence(seed);
  test_chunked_matches_oneshot(seed + 1);
  test_shared_state_code_switch(seed + 2);
  for (int j = 0; j < N_FAM; ++j) {
    test_round_trip(seed + 10 + (uint64_t)j, j);
  }
  test_args();
  return test_summary("multi_encode");
}
