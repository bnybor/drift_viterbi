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
 * Tests for the encoder and streaming decoder: chunked encoding, clean and
 * noisy stream decoding (erasures, indels, re-anchoring, blind acquisition),
 * the standard presets, lock probability, and argument handling.
 */

#include "dv_test_util.h"

/* rate-1/5, K=5 code (matches the Python reference). */
static const unsigned int GENERATORS[] = {037, 033, 025, 027, 035};
#define K 5
#define N_GEN ((int)(sizeof(GENERATORS) / sizeof(GENERATORS[0])))

static void test_version(void) {
  printf("test_version\n");
  const char *v = drift_viterbi_version();
  REQUIRE("version non-null", v != NULL);
  check("version non-empty", strlen(v) > 0);
  printf("  drift_viterbi version: %s\n", v);
}

/* The streaming encoder is stateful: encoding in chunks (plus a final flush)
 * must match encoding the whole message in one call, and end in state 0. */
static void test_encode_stream(void) {
  printf("test_encode_stream\n");
  dv_code *code = dv_code_create(K, GENERATORS, N_GEN);
  REQUIRE("code created", code != NULL);
  check("code n", dv_code_n(code) == N_GEN);
  check("code k", dv_code_k(code) == K);

  const int n_info = 200;
  uint8_t *msg = malloc((size_t)n_info);
  for (int i = 0; i < n_info; ++i) msg[i] = (uint8_t)((i * 7 + 3) & 1);

  int data_len = n_info * N_GEN;
  int total_len = data_len + (K - 1) * N_GEN; /* + flush */

  uint8_t *ref = malloc((size_t)total_len);
  int rstate = 0, ri = 0;
  ri += dv_code_encode(code, msg, n_info, &rstate, ref);
  ri += dv_code_encode_flush(code, &rstate, ref + ri);
  check("one-shot length and end state", ri == total_len && rstate == 0);

  uint8_t *out = malloc((size_t)total_len);
  int state = 0, oi = 0;
  const int n1 = 80, n2 = n_info - n1;
  int w = dv_code_encode(code, msg, n1, &state, out);
  check("chunk 1 length", w == n1 * N_GEN);
  oi += w;
  w = dv_code_encode(code, msg + n1, n2, &state, out + oi);
  check("chunk 2 length", w == n2 * N_GEN);
  oi += w;
  w = dv_code_encode_flush(code, &state, out + oi);
  check("flush length", w == (K - 1) * N_GEN);
  oi += w;

  check("chunked total length and end state", oi == total_len && state == 0);
  check("chunked matches one-shot", memcmp(out, ref, (size_t)total_len) == 0);
  printf("  %d bits, chunked matches one-shot, end state=%d\n", oi, state);

  free(out);
  free(ref);
  free(msg);
  dv_code_destroy(code);
}

/* A clean continuously-encoded stream, pushed through the sliding-window
 * decoder in small chunks, must come back out exactly. */
static void test_stream_clean(void) {
  printf("test_stream_clean\n");
  dv_code *code = dv_code_create(K, GENERATORS, N_GEN);
  REQUIRE("code created", code != NULL);

  const int n_info = 300;
  uint8_t *msg = malloc((size_t)n_info);
  for (int i = 0; i < n_info; ++i) msg[i] = (uint8_t)((i * 7 + 3) & 1);

  int clen = n_info * N_GEN;
  uint8_t *coded = malloc((size_t)clen);
  int st = 0;
  check("encode length", dv_code_encode(code, msg, n_info, &st, coded) == clen);

  dv_stream_decoder *sd = make_decoder(code, 40, 4, 0.01, 0.01, 0.01, 0.0);
  REQUIRE("decoder created", sd != NULL);

  uint8_t *outbuf = malloc((size_t)n_info + 16);
  int got = stream_decode_all(sd, coded, clen, outbuf, n_info + 16);

  int errors = 0;
  for (int i = 0; i < n_info; ++i) errors += (outbuf[i] != msg[i]);
  printf("  decoded %d bits (msg %d), errors=%d\n", got, n_info, errors);
  check("clean stream recovered exactly", got == n_info && errors == 0);

  dv_stream_decoder_destroy(sd);
  free(outbuf);
  free(coded);
  free(msg);
  dv_code_destroy(code);
}

/* A clean stream with a burst of received bits marked erased: the decoder
 * abstains on them and still recovers the message. */
static void test_stream_erasures(void) {
  printf("test_stream_erasures\n");
  dv_code *code = dv_code_create(K, GENERATORS, N_GEN);
  REQUIRE("code created", code != NULL);

  const int n_info = 300;
  uint8_t *msg = malloc((size_t)n_info);
  for (int i = 0; i < n_info; ++i) msg[i] = (uint8_t)((i * 7 + 3) & 1);

  int clen = n_info * N_GEN;
  uint8_t *rx = malloc((size_t)clen);
  int st = 0;
  check("encode length", dv_code_encode(code, msg, n_info, &st, rx) == clen);
  for (int i = 700; i < 716; ++i) rx[i] = DV_ERASURE; /* 16-bit burst */

  dv_stream_decoder *sd = make_decoder(code, 40, 4, 0.01, 0.01, 0.01, 0.05);
  REQUIRE("decoder created", sd != NULL);

  uint8_t *outbuf = malloc((size_t)n_info + 16);
  int got = stream_decode_all(sd, rx, clen, outbuf, n_info + 16);

  int errors = 0;
  for (int i = 0; i < n_info; ++i) errors += (outbuf[i] != msg[i]);
  printf("  16-bit erasure burst, decoded %d bits, errors=%d\n", got, errors);
  check("recovered through erasure burst", got == n_info && errors == 0);

  dv_stream_decoder_destroy(sd);
  free(outbuf);
  free(rx);
  free(msg);
  dv_code_destroy(code);
}

/* A long stream with periodic deletions: cumulative drift grows far past
 * max_drift, so only re-anchoring keeps the decoder locked. */
static void test_stream_reanchor(void) {
  printf("test_stream_reanchor\n");
  dv_code *code = dv_code_create(K, GENERATORS, N_GEN);
  REQUIRE("code created", code != NULL);

  const int n_info = 400;
  uint8_t *msg = malloc((size_t)n_info);
  for (int i = 0; i < n_info; ++i) msg[i] = (uint8_t)((i * 11 + 5) & 1);

  int clen = n_info * N_GEN; /* 2000 coded bits */
  uint8_t *coded = malloc((size_t)clen);
  int st = 0;
  check("encode length", dv_code_encode(code, msg, n_info, &st, coded) == clen);

  /* Delete one coded bit every 100 -> ~20 deletions, cumulative drift ~ -20. */
  const int del_period = 100;
  uint8_t *rx = malloc((size_t)clen);
  int rl = 0, ndel = 0;
  for (int i = 0; i < clen; ++i) {
    if ((i + 1) % del_period == 0) {
      ndel++;
      continue;
    }
    rx[rl++] = coded[i];
  }

  dv_stream_decoder *sd = make_decoder(code, 48, 6, 0.01, 0.01, 0.01, 0.0);
  REQUIRE("decoder created", sd != NULL);

  int cap = n_info + 32;
  uint8_t *outbuf = malloc((size_t)cap);
  int got = stream_decode_all(sd, rx, rl, outbuf, cap);

  int cmp = got < n_info ? got : n_info;
  int errors = 0;
  for (int i = 0; i < cmp; ++i) errors += (outbuf[i] != msg[i]);
  printf(
      "  %d deletions (cum drift -%d, max_drift 6), decoded %d/%d bits, "
      "errors=%d\n",
      ndel, ndel, got, n_info, errors);

  check("output length within +/-2", got >= n_info - 2 && got <= n_info + 2);
  /* bit-level alignment recovers these deletions exactly */
  check("deletions recovered exactly", errors == 0);

  dv_stream_decoder_destroy(sd);
  free(outbuf);
  free(rx);
  free(coded);
  free(msg);
  dv_code_destroy(code);
}

/* Indels placed in the MIDDLE of coded groups, not at group boundaries. The
 * bit-level alignment tracks an indel at any bit position, so an otherwise
 * clean stream is recovered essentially perfectly - the group holding the indel
 * is no longer smeared into a burst of substitution errors. */
static void test_stream_midgroup_indel(void) {
  printf("test_stream_midgroup_indel\n");
  dv_code *code = dv_code_create(K, GENERATORS, N_GEN); /* group size N_GEN=5 */
  REQUIRE("code created", code != NULL);

  const int n_info = 400;
  uint8_t *msg = malloc((size_t)n_info);
  for (int i = 0; i < n_info; ++i) msg[i] = (uint8_t)((i * 11 + 5) & 1);

  int clen = n_info * N_GEN;
  uint8_t *coded = malloc((size_t)clen);
  int st = 0;
  check("encode length", dv_code_encode(code, msg, n_info, &st, coded) == clen);

  /* Drop a bit at phase 2 and insert one at phase 62 of every 120 coded bits.
   * 120 is a multiple of N_GEN, so both land at offset 2 within a group - mid
   * group, never on a boundary. */
  uint8_t *rx = malloc((size_t)clen + 64);
  int rl = 0, ndel = 0, nins = 0;
  for (int i = 0; i < clen; ++i) {
    const int phase = i % 120;
    if (phase == 2) { /* mid-group deletion */
      ndel++;
      continue;
    }
    if (phase == 62) { /* mid-group insertion of a spurious bit */
      rx[rl++] = (uint8_t)((i >> 1) & 1);
      nins++;
    }
    rx[rl++] = coded[i];
  }

  dv_stream_decoder *sd = make_decoder(code, 48, 6, 0.01, 0.01, 0.01, 0.0);
  REQUIRE("decoder created", sd != NULL);

  int cap = n_info + 32;
  uint8_t *outbuf = malloc((size_t)cap);
  int got = stream_decode_all(sd, rx, rl, outbuf, cap);

  int cmp = got < n_info ? got : n_info;
  int errors = 0;
  for (int i = 0; i < cmp; ++i) errors += (outbuf[i] != msg[i]);
  printf(
      "  %d deletions + %d insertions mid-group, decoded %d/%d bits, "
      "errors=%d\n",
      ndel, nins, got, n_info, errors);

  check("output length within +/-2", got >= n_info - 2 && got <= n_info + 2);
  check("mid-group indels recovered exactly", errors == 0);

  dv_stream_decoder_destroy(sd);
  free(outbuf);
  free(rx);
  free(coded);
  free(msg);
  dv_code_destroy(code);
}

/* A built-in standard code (K=7 rate 1/2) encodes and streams cleanly. */
static void test_standard_code(void) {
  printf("test_standard_code\n");
  dv_code *code = dv_code_create_standard(DV_CODE_K7_RATE_1_2);
  REQUIRE("code created", code != NULL);
  check("standard code n", dv_code_n(code) == 2);
  check("standard code k", dv_code_k(code) == 7);

  const int n_info = 150;
  uint8_t *msg = malloc((size_t)n_info);
  for (int i = 0; i < n_info; ++i) msg[i] = (uint8_t)((i * 5 + 1) & 1);

  int clen = n_info * 2;
  uint8_t *coded = malloc((size_t)clen);
  int st = 0;
  check("encode length", dv_code_encode(code, msg, n_info, &st, coded) == clen);

  dv_stream_decoder *sd = make_decoder(code, 48, 4, 0.01, 0.01, 0.01, 0.0);
  REQUIRE("decoder created", sd != NULL);

  uint8_t *outbuf = malloc((size_t)n_info + 16);
  int got = stream_decode_all(sd, coded, clen, outbuf, n_info + 16);

  int errors = 0;
  for (int i = 0; i < n_info; ++i) errors += (outbuf[i] != msg[i]);
  printf("  K7R12 decoded %d bits, errors=%d\n", got, errors);
  check("standard code recovered exactly", got == n_info && errors == 0);

  check("bad preset rejected",
        dv_code_create_standard((dv_standard_code)999) == NULL);

  dv_stream_decoder_destroy(sd);
  free(outbuf);
  free(coded);
  free(msg);
  dv_code_destroy(code);
}

/* Tap into a stream partway through, where the encoder state is unknown. The
 * decoder blind-acquires (always) and recovers the rest of the message after a
 * short transient. */
static void test_blind_acquisition(void) {
  printf("test_blind_acquisition\n");
  dv_code *code = dv_code_create(K, GENERATORS, N_GEN);
  REQUIRE("code created", code != NULL);

  const int n_info = 400;
  uint8_t *msg = malloc((size_t)n_info);
  for (int i = 0; i < n_info; ++i) msg[i] = (uint8_t)((i * 13 + 7) & 1);

  int clen = n_info * N_GEN;
  uint8_t *coded = malloc((size_t)clen);
  int st = 0;
  check("encode length", dv_code_encode(code, msg, n_info, &st, coded) == clen);

  /* Start decoding from input step 100 (a group boundary), where the encoder
   * state is some unknown non-zero value. */
  const int splice_step = 100;
  const uint8_t *rx = coded + splice_step * N_GEN;
  int rl = clen - splice_step * N_GEN;
  int avail = n_info - splice_step; /* input bits available */

  dv_stream_decoder *sd = make_decoder(code, 40, 4, 0.01, 0.01, 0.01, 0.0);
  REQUIRE("decoder created", sd != NULL);

  int cap = avail + 16;
  uint8_t *out = malloc((size_t)cap);
  int got = stream_decode_all(sd, rx, rl, out, cap);

  /* After the acquisition transient, decoded bit j corresponds to msg[splice
   * + j] and should match exactly on this clean channel. */
  const int warmup = 120;
  int cmp = got < avail ? got : avail;
  int errors = 0, counted = 0;
  for (int j = warmup; j < cmp; ++j) {
    errors += (out[j] != msg[splice_step + j]);
    counted++;
  }
  printf("  spliced at step %d, decoded %d bits, post-warmup errors=%d/%d\n",
         splice_step, got, errors, counted);
  check("post-warmup bits counted", counted > 0);
  check("post-warmup errors zero", errors == 0);

  dv_stream_decoder_destroy(sd);
  free(out);
  free(coded);
  free(msg);
  dv_code_destroy(code);
}

/* Minimal settings: leave max_drift (and the indel probabilities) at 0, so the
 * decoder corrects bit flips only. A few scattered flips are still fixed. */
static void test_stream_flips_only(void) {
  printf("test_stream_flips_only\n");
  dv_code *code = dv_code_create(K, GENERATORS, N_GEN);
  REQUIRE("code created", code != NULL);

  const int n_info = 300;
  uint8_t *msg = malloc((size_t)n_info);
  for (int i = 0; i < n_info; ++i) msg[i] = (uint8_t)((i * 7 + 3) & 1);

  int clen = n_info * N_GEN;
  uint8_t *rx = malloc((size_t)clen);
  int st = 0;
  check("encode length", dv_code_encode(code, msg, n_info, &st, rx) == clen);

  int flips[] = {123, 400, 777, 1100, 1450};
  const int nflip = (int)(sizeof(flips) / sizeof(flips[0]));
  for (int i = 0; i < nflip; ++i) rx[flips[i]] ^= 1;

  dv_stream_params params = {
      .decision_depth = 40,
      .p_sub = 0.02,
  };
  dv_stream_decoder *sd = dv_stream_decoder_create(code, &params);
  REQUIRE("decoder created", sd != NULL);

  uint8_t *outbuf = malloc((size_t)n_info + 16);
  int got = stream_decode_all(sd, rx, clen, outbuf, n_info + 16);

  int errors = 0;
  for (int i = 0; i < n_info; ++i) errors += (outbuf[i] != msg[i]);
  printf("  %d flips, decoded %d bits, errors=%d\n", nflip, got, errors);
  check("flips corrected", got == n_info && errors == 0);

  dv_stream_decoder_destroy(sd);
  free(outbuf);
  free(rx);
  free(msg);
  dv_code_destroy(code);
}

/* Every standard preset (the four defaults plus their two alternates each) must
 * create, report the right rate/K, and round-trip a clean stream exactly. */
static void test_all_presets(void) {
  printf("test_all_presets\n");
  struct {
    dv_standard_code code;
    int n, k;
  } presets[] = {
      {DV_CODE_K3_RATE_1_2, 2, 3},      {DV_CODE_K3_RATE_1_2_ALT1, 2, 3},
      {DV_CODE_K3_RATE_1_2_ALT2, 2, 3}, {DV_CODE_K7_RATE_1_2, 2, 7},
      {DV_CODE_K7_RATE_1_2_ALT1, 2, 7}, {DV_CODE_K7_RATE_1_2_ALT2, 2, 7},
      {DV_CODE_K7_RATE_1_3, 3, 7},      {DV_CODE_K7_RATE_1_3_ALT1, 3, 7},
      {DV_CODE_K7_RATE_1_3_ALT2, 3, 7}, {DV_CODE_K5_RATE_1_5, 5, 5},
      {DV_CODE_K5_RATE_1_5_ALT1, 5, 5}, {DV_CODE_K5_RATE_1_5_ALT2, 5, 5},
  };
  const int np = (int)(sizeof(presets) / sizeof(presets[0]));

  int all_ok = 1;
  for (int idx = 0; idx < np; ++idx) {
    dv_code *code = dv_code_create_standard(presets[idx].code);
    REQUIRE("preset created", code != NULL);
    if (dv_code_n(code) != presets[idx].n ||
        dv_code_k(code) != presets[idx].k) {
      all_ok = 0;
    }

    const int n_info = 200;
    uint8_t msg[200];
    for (int i = 0; i < n_info; ++i) msg[i] = (uint8_t)((i * 5 + 1) & 1);
    int clen = n_info * presets[idx].n;
    uint8_t *coded = malloc((size_t)clen);
    int st = 0;
    if (dv_code_encode(code, msg, n_info, &st, coded) != clen) {
      all_ok = 0;
    }

    dv_stream_decoder *sd =
        make_decoder(code, 8 * presets[idx].k, 4, 0.01, 0.01, 0.01, 0.0);
    REQUIRE("preset decoder created", sd != NULL);
    int cap = n_info + 32;
    uint8_t *out = malloc((size_t)cap);
    int got = stream_decode_all(sd, coded, clen, out, cap);
    int cmp = got < n_info ? got : n_info, errors = 0;
    for (int i = 0; i < cmp; ++i) errors += (out[i] != msg[i]);
    if (got != n_info || errors != 0) {
      all_ok = 0;
    }

    dv_stream_decoder_destroy(sd);
    free(out);
    free(coded);
    dv_code_destroy(code);
  }
  printf("  %d presets create, report rate/K, and round-trip cleanly\n", np);
  check("all presets round-trip cleanly", all_ok);
}

/* The lock-probability output rises to ~1 on a clean coded stream (the decoder
 * is synchronized) and stays low on random, non-coded input (it never locks).
 * A NULL lock pointer is also accepted. */
static void test_lock_probability(void) {
  printf("test_lock_probability\n");
  dv_code *code = dv_code_create(K, GENERATORS, N_GEN);
  REQUIRE("code created", code != NULL);

  const int n_info = 600;
  uint8_t *msg = malloc((size_t)n_info);
  for (int i = 0; i < n_info; ++i) msg[i] = (uint8_t)((i * 7 + 3) & 1);

  int clen = n_info * N_GEN;
  uint8_t *coded = malloc((size_t)clen);
  int st = 0;
  check("encode length", dv_code_encode(code, msg, n_info, &st, coded) == clen);

  int cap = n_info + 32;
  uint8_t *out = malloc((size_t)cap);
  double *lock = malloc((size_t)cap * sizeof(double));

  /* Clean coded stream: feed it all at once, capturing per-bit lock prob. */
  dv_stream_decoder *sd = make_decoder(code, 48, 4, 0.01, 0.01, 0.01, 0.0);
  REQUIRE("decoder created", sd != NULL);
  int got = dv_stream_decode(sd, coded, clen, out, lock, cap);
  REQUIRE("clean decode produced output", got > 0);
  double clean_sum = 0.0;
  int prob_in_range = 1;
  for (int i = 0; i < got; ++i) {
    if (!(lock[i] > 0.0 && lock[i] <= 1.0 + 1e-9)) prob_in_range = 0;
    if (i >= got / 2) clean_sum += lock[i];
  }
  check("lock values are probabilities", prob_in_range);
  double clean_mean = clean_sum / (got - got / 2);
  dv_stream_decoder_destroy(sd);

  /* Random, non-coded input (a deterministic LCG of bits): the decoder cannot
   * lock onto a codeword path, so the probability stays low. */
  uint8_t *rnd = malloc((size_t)clen);
  uint64_t lcg = 0x1234567u;
  for (int i = 0; i < clen; ++i) {
    lcg = lcg * 6364136223846793005ULL + 1u;
    rnd[i] = (uint8_t)((lcg >> 40) & 1u);
  }
  sd = make_decoder(code, 48, 4, 0.01, 0.01, 0.01, 0.0);
  REQUIRE("decoder created", sd != NULL);
  int got2 = dv_stream_decode(sd, rnd, clen, out, lock, cap);
  REQUIRE("random decode produced output", got2 > 0);
  double rnd_sum = 0.0;
  for (int i = got2 / 2; i < got2; ++i) rnd_sum += lock[i];
  double rnd_mean = rnd_sum / (got2 - got2 / 2);
  dv_stream_decoder_destroy(sd);

  printf("  clean mean=%.3f, random mean=%.3f\n", clean_mean, rnd_mean);
  check_gt("locked on coded data", clean_mean, 0.85);
  check_lt("never locks on noise", rnd_mean, 0.6);
  check_gt("clean vs random margin", clean_mean - rnd_mean, 0.3);

  /* A NULL lock pointer must be accepted and decode normally. */
  sd = make_decoder(code, 48, 4, 0.01, 0.01, 0.01, 0.0);
  REQUIRE("decoder created", sd != NULL);
  check("NULL lock pointer accepted",
        dv_stream_decode(sd, coded, clen, out, NULL, cap) > 0);
  dv_stream_decoder_destroy(sd);

  free(lock);
  free(out);
  free(rnd);
  free(coded);
  free(msg);
  dv_code_destroy(code);

  /* The two rate-1/2 presets produce structured (not random) streams of the
   * same rate, but a stream coded with one must NOT look locked to the other's
   * decoder - only the matching code locks. */
  dv_code *k3 = dv_code_create_standard(DV_CODE_K3_RATE_1_2);
  dv_code *k7 = dv_code_create_standard(DV_CODE_K7_RATE_1_2);
  REQUIRE("k3 and k7 created", k3 != NULL && k7 != NULL);

  uint8_t lmsg[600];
  for (int i = 0; i < 600; ++i) lmsg[i] = (uint8_t)((i * 7 + 3) & 1);
  double k3_k3 = decoder_lock_mean(k3, k3, lmsg, 600, 48);
  double k7_k7 = decoder_lock_mean(k7, k7, lmsg, 600, 48);
  double k3_k7 = decoder_lock_mean(k3, k7, lmsg, 600, 48);
  double k7_k3 = decoder_lock_mean(k7, k3, lmsg, 600, 48);
  printf("  match k3=%.3f k7=%.3f, cross k3->k7=%.3f k7->k3=%.3f\n", k3_k3,
         k7_k7, k3_k7, k7_k3);

  check_gt("k3 locks on its own stream", k3_k3, 0.85);
  check_gt("k7 locks on its own stream", k7_k7, 0.85);
  check_lt("k3 does not lock on k7", k3_k7, 0.75);
  check_lt("k7 does not lock on k3", k7_k3, 0.75);
  check_gt("k3 self vs cross margin", k3_k3 - k3_k7, 0.3);
  check_gt("k7 self vs cross margin", k7_k7 - k7_k3, 0.3);

  dv_code_destroy(k3);
  dv_code_destroy(k7);
}

/* Within each (K, rate) family the default and its two alternates are picked to
 * be mutually distinguishable: each locks onto its own stream but not onto a
 * sibling's. (Across families - different K or rate - this is NOT guaranteed;
 * see the comment on dv_standard_code.) */
static void test_cross_lock_within_family(void) {
  printf("test_cross_lock_within_family\n");
  struct {
    const char *name;
    dv_standard_code v[3];
  } fam[] = {
      {"K3_R1_2",
       {DV_CODE_K3_RATE_1_2, DV_CODE_K3_RATE_1_2_ALT1,
        DV_CODE_K3_RATE_1_2_ALT2}},
      {"K7_R1_2",
       {DV_CODE_K7_RATE_1_2, DV_CODE_K7_RATE_1_2_ALT1,
        DV_CODE_K7_RATE_1_2_ALT2}},
      {"K7_R1_3",
       {DV_CODE_K7_RATE_1_3, DV_CODE_K7_RATE_1_3_ALT1,
        DV_CODE_K7_RATE_1_3_ALT2}},
      {"K5_R1_5",
       {DV_CODE_K5_RATE_1_5, DV_CODE_K5_RATE_1_5_ALT1,
        DV_CODE_K5_RATE_1_5_ALT2}},
  };
  const int nf = (int)(sizeof(fam) / sizeof(fam[0]));

  uint8_t msg[600];
  for (int i = 0; i < 600; ++i) msg[i] = (uint8_t)((i * 7 + 3) & 1);

  double max_cross = 0.0, min_self = 1.0;
  for (int f = 0; f < nf; ++f) {
    dv_code *code[3];
    for (int i = 0; i < 3; ++i) {
      code[i] = dv_code_create_standard(fam[f].v[i]);
      REQUIRE("family code created", code[i] != NULL);
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        double m = decoder_lock_mean(code[i], code[j], msg, 600, 48);
        if (i == j) {
          if (m < min_self) min_self = m;
        } else {
          if (m > max_cross) max_cross = m;
        }
      }
    }
    for (int i = 0; i < 3; ++i) dv_code_destroy(code[i]);
  }
  printf("  max sibling=%.3f, min self=%.3f\n", max_cross, min_self);
  check_gt("each code locks onto its own stream", min_self, 0.9);
  check_lt("no code locks onto a sibling's", max_cross, 0.75);
}

static void test_error_paths(void) {
  printf("test_error_paths\n");
  /* Code creation rejects bad arguments by returning NULL. */
  check("create rejects K<2", dv_code_create(1, GENERATORS, N_GEN) == NULL);
  check("create rejects n<1", dv_code_create(K, GENERATORS, 0) == NULL);
  check("dv_code_n(NULL) is -1", dv_code_n(NULL) == -1);

  dv_code *code = dv_code_create(K, GENERATORS, N_GEN);
  REQUIRE("code created", code != NULL);

  /* Encoder rejects an out-of-range carried-in state. */
  uint8_t bit = DV_TRUE, obuf[N_GEN];
  int badstate = 1 << 20;
  check("encode rejects bad state",
        dv_code_encode(code, &bit, 1, &badstate, obuf) == DV_ERR_ARG);

  /* Decoder creation rejects bad settings by returning NULL. */
  dv_stream_params ok = {
      .decision_depth = 40,
      .max_drift = 4,
      .p_sub = 0.01,
      .p_ins = 0.01,
      .p_del = 0.01,
  };
  check("decoder rejects null code",
        dv_stream_decoder_create(NULL, &ok) == NULL);
  check("decoder rejects null params",
        dv_stream_decoder_create(code, NULL) == NULL);

  dv_stream_params p;
  p = ok;
  p.decision_depth = 0;
  check("decoder rejects depth 0", dv_stream_decoder_create(code, &p) == NULL);
  p = ok;
  p.max_drift = -1;
  check("decoder rejects negative drift",
        dv_stream_decoder_create(code, &p) == NULL);
  p = ok;
  p.p_sub = 0.0;
  check("decoder rejects p_sub 0", dv_stream_decoder_create(code, &p) == NULL);
  p = ok;
  p.p_ins = p.p_del = 0.6;
  check("decoder rejects p_ins+p_del>=1",
        dv_stream_decoder_create(code, &p) == NULL);
  p = ok;
  p.p_erase = 1.0;
  check("decoder rejects p_erase 1",
        dv_stream_decoder_create(code, &p) == NULL);
  /* max_drift > 0 needs insertion/deletion probabilities. */
  p = ok;
  p.p_ins = p.p_del = 0.0;
  check("decoder rejects drift without indel probs",
        dv_stream_decoder_create(code, &p) == NULL);

  dv_stream_decoder *sd = dv_stream_decoder_create(code, &ok);
  REQUIRE("decoder created", sd != NULL);

  /* Streaming decode argument checks. */
  uint8_t out8 = 0;
  check("decode rejects null decoder",
        dv_stream_decode(NULL, &bit, 1, &out8, NULL, 1) == DV_ERR_ARG);
  check("decode rejects null input",
        dv_stream_decode(sd, NULL, 1, &out8, NULL, 1) == DV_ERR_ARG);
  check("flush rejects null decoder",
        dv_stream_decode_flush(NULL, &out8, 1) == DV_ERR_ARG);

  dv_stream_decoder_destroy(sd);
  dv_code_destroy(code);
}

int main(void) {
  test_version();
  test_encode_stream();
  test_stream_clean();
  test_stream_erasures();
  test_stream_reanchor();
  test_stream_midgroup_indel();
  test_standard_code();
  test_all_presets();
  test_blind_acquisition();
  test_stream_flips_only();
  test_lock_probability();
  test_cross_lock_within_family();
  test_error_paths();
  return test_summary("decoder");
}
