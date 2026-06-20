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

#include <drift_viterbi/drift_viterbi.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* rate-1/5, K=5 code (matches the Python reference). */
static const unsigned int GENERATORS[] = {037, 033, 025, 027, 035};
#define K 5
#define N_GEN ((int)(sizeof(GENERATORS) / sizeof(GENERATORS[0])))

static void test_version(void) {
  const char *v = drift_viterbi_version();
  assert(v != NULL && strlen(v) > 0);
  printf("drift_viterbi version: %s\n", v);
}

/* Build a decoder from positional settings (keeps tests concise). */
static dv_stream_decoder *make_decoder(const dv_code *code, int depth,
                                       int drift, double p_sub, double p_ins,
                                       double p_del, double p_erase) {
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
static int stream_decode_all(dv_stream_decoder *sd, const uint8_t *rx, int rl,
                             uint8_t *out, int cap) {
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

/* Encode a fixed message with `enc`, decode it with a fresh decoder for `dec`
 * (both must have the same output rate), and return the mean lock probability
 * over the locked second half. When enc != dec this measures whether one code's
 * stream is mistaken for the other's. */
static double lock_mean_cross(const dv_code *enc, const dv_code *dec) {
  enum { N_INFO = 600 };
  uint8_t msg[N_INFO];
  for (int i = 0; i < N_INFO; ++i) msg[i] = (uint8_t)((i * 7 + 3) & 1);

  int clen = N_INFO * dv_code_n(enc);
  uint8_t *coded = malloc((size_t)clen);
  int st = 0;
  assert(dv_code_encode(enc, msg, N_INFO, &st, coded) == clen);

  dv_stream_decoder *sd = make_decoder(dec, 48, 4, 0.01, 0.01, 0.01, 0.0);
  assert(sd != NULL);
  int cap = N_INFO + 64;
  uint8_t *out = malloc((size_t)cap);
  double *lock = malloc((size_t)cap * sizeof(double));
  int got = dv_stream_decode(sd, coded, clen, out, lock, cap);
  assert(got > 0);

  double sum = 0.0;
  for (int i = got / 2; i < got; ++i) sum += lock[i];
  double m = sum / (got - got / 2);

  dv_stream_decoder_destroy(sd);
  free(lock);
  free(out);
  free(coded);
  return m;
}

/* The streaming encoder is stateful: encoding in chunks (plus a final flush)
 * must match encoding the whole message in one call, and end in state 0. */
static void test_encode_stream(void) {
  dv_code *code = dv_code_create(K, GENERATORS, N_GEN);
  assert(code != NULL);
  assert(dv_code_n(code) == N_GEN);
  assert(dv_code_k(code) == K);

  const int n_info = 200;
  uint8_t *msg = malloc((size_t)n_info);
  for (int i = 0; i < n_info; ++i) msg[i] = (uint8_t)((i * 7 + 3) & 1);

  int data_len = n_info * N_GEN;
  int total_len = data_len + (K - 1) * N_GEN; /* + flush */

  uint8_t *ref = malloc((size_t)total_len);
  int rstate = 0, ri = 0;
  ri += dv_code_encode(code, msg, n_info, &rstate, ref);
  ri += dv_code_encode_flush(code, &rstate, ref + ri);
  assert(ri == total_len && rstate == 0);

  uint8_t *out = malloc((size_t)total_len);
  int state = 0, oi = 0;
  const int n1 = 80, n2 = n_info - n1;
  int w = dv_code_encode(code, msg, n1, &state, out);
  assert(w == n1 * N_GEN);
  oi += w;
  w = dv_code_encode(code, msg + n1, n2, &state, out + oi);
  assert(w == n2 * N_GEN);
  oi += w;
  w = dv_code_encode_flush(code, &state, out + oi);
  assert(w == (K - 1) * N_GEN);
  oi += w;

  assert(oi == total_len && state == 0);
  assert(memcmp(out, ref, (size_t)total_len) == 0);
  printf("encode stream: %d bits, chunked matches one-shot, end state=%d\n", oi,
         state);

  free(out);
  free(ref);
  free(msg);
  dv_code_destroy(code);
}

/* A clean continuously-encoded stream, pushed through the sliding-window
 * decoder in small chunks, must come back out exactly. */
static void test_stream_clean(void) {
  dv_code *code = dv_code_create(K, GENERATORS, N_GEN);
  assert(code != NULL);

  const int n_info = 300;
  uint8_t *msg = malloc((size_t)n_info);
  for (int i = 0; i < n_info; ++i) msg[i] = (uint8_t)((i * 7 + 3) & 1);

  int clen = n_info * N_GEN;
  uint8_t *coded = malloc((size_t)clen);
  int st = 0;
  assert(dv_code_encode(code, msg, n_info, &st, coded) == clen);

  dv_stream_decoder *sd = make_decoder(code, 40, 4, 0.01, 0.01, 0.01, 0.0);
  assert(sd != NULL);

  uint8_t *outbuf = malloc((size_t)n_info + 16);
  int got = stream_decode_all(sd, coded, clen, outbuf, n_info + 16);

  int errors = 0;
  for (int i = 0; i < n_info; ++i) errors += (outbuf[i] != msg[i]);
  printf("stream clean: decoded %d bits (msg %d), errors=%d\n", got, n_info,
         errors);
  assert(got == n_info && errors == 0);

  dv_stream_decoder_destroy(sd);
  free(outbuf);
  free(coded);
  free(msg);
  dv_code_destroy(code);
}

/* A clean stream with a burst of received bits marked erased: the decoder
 * abstains on them and still recovers the message. */
static void test_stream_erasures(void) {
  dv_code *code = dv_code_create(K, GENERATORS, N_GEN);
  assert(code != NULL);

  const int n_info = 300;
  uint8_t *msg = malloc((size_t)n_info);
  for (int i = 0; i < n_info; ++i) msg[i] = (uint8_t)((i * 7 + 3) & 1);

  int clen = n_info * N_GEN;
  uint8_t *rx = malloc((size_t)clen);
  int st = 0;
  assert(dv_code_encode(code, msg, n_info, &st, rx) == clen);
  for (int i = 700; i < 716; ++i) rx[i] = DV_ERASURE; /* 16-bit burst */

  dv_stream_decoder *sd = make_decoder(code, 40, 4, 0.01, 0.01, 0.01, 0.05);
  assert(sd != NULL);

  uint8_t *outbuf = malloc((size_t)n_info + 16);
  int got = stream_decode_all(sd, rx, clen, outbuf, n_info + 16);

  int errors = 0;
  for (int i = 0; i < n_info; ++i) errors += (outbuf[i] != msg[i]);
  printf("stream erasures: 16-bit burst, decoded %d bits, errors=%d\n", got,
         errors);
  assert(got == n_info && errors == 0);

  dv_stream_decoder_destroy(sd);
  free(outbuf);
  free(rx);
  free(msg);
  dv_code_destroy(code);
}

/* A long stream with periodic deletions: cumulative drift grows far past
 * max_drift, so only re-anchoring keeps the decoder locked. */
static void test_stream_reanchor(void) {
  dv_code *code = dv_code_create(K, GENERATORS, N_GEN);
  assert(code != NULL);

  const int n_info = 400;
  uint8_t *msg = malloc((size_t)n_info);
  for (int i = 0; i < n_info; ++i) msg[i] = (uint8_t)((i * 11 + 5) & 1);

  int clen = n_info * N_GEN; /* 2000 coded bits */
  uint8_t *coded = malloc((size_t)clen);
  int st = 0;
  assert(dv_code_encode(code, msg, n_info, &st, coded) == clen);

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
  assert(sd != NULL);

  int cap = n_info + 32;
  uint8_t *outbuf = malloc((size_t)cap);
  int got = stream_decode_all(sd, rx, rl, outbuf, cap);

  int cmp = got < n_info ? got : n_info;
  int errors = 0;
  for (int i = 0; i < cmp; ++i) errors += (outbuf[i] != msg[i]);
  printf(
      "stream reanchor: %d deletions (cum drift -%d, max_drift 6), "
      "decoded %d/%d bits, errors=%d\n",
      ndel, ndel, got, n_info, errors);

  assert(got >= n_info - 2 && got <= n_info + 2);
  assert(errors == 0); /* bit-level alignment recovers these deletions exactly */

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
  dv_code *code = dv_code_create(K, GENERATORS, N_GEN); /* group size N_GEN=5 */
  assert(code != NULL);

  const int n_info = 400;
  uint8_t *msg = malloc((size_t)n_info);
  for (int i = 0; i < n_info; ++i) msg[i] = (uint8_t)((i * 11 + 5) & 1);

  int clen = n_info * N_GEN;
  uint8_t *coded = malloc((size_t)clen);
  int st = 0;
  assert(dv_code_encode(code, msg, n_info, &st, coded) == clen);

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
  assert(sd != NULL);

  int cap = n_info + 32;
  uint8_t *outbuf = malloc((size_t)cap);
  int got = stream_decode_all(sd, rx, rl, outbuf, cap);

  int cmp = got < n_info ? got : n_info;
  int errors = 0;
  for (int i = 0; i < cmp; ++i) errors += (outbuf[i] != msg[i]);
  printf(
      "stream mid-group indel: %d deletions + %d insertions mid-group, "
      "decoded %d/%d bits, errors=%d\n",
      ndel, nins, got, n_info, errors);

  assert(got >= n_info - 2 && got <= n_info + 2);
  assert(errors == 0);

  dv_stream_decoder_destroy(sd);
  free(outbuf);
  free(rx);
  free(coded);
  free(msg);
  dv_code_destroy(code);
}

/* A built-in standard code (K=7 rate 1/2) encodes and streams cleanly. */
static void test_standard_code(void) {
  dv_code *code = dv_code_create_standard(DV_CODE_K7_RATE_1_2);
  assert(code != NULL);
  assert(dv_code_n(code) == 2);
  assert(dv_code_k(code) == 7);

  const int n_info = 150;
  uint8_t *msg = malloc((size_t)n_info);
  for (int i = 0; i < n_info; ++i) msg[i] = (uint8_t)((i * 5 + 1) & 1);

  int clen = n_info * 2;
  uint8_t *coded = malloc((size_t)clen);
  int st = 0;
  assert(dv_code_encode(code, msg, n_info, &st, coded) == clen);

  dv_stream_decoder *sd = make_decoder(code, 48, 4, 0.01, 0.01, 0.01, 0.0);
  assert(sd != NULL);

  uint8_t *outbuf = malloc((size_t)n_info + 16);
  int got = stream_decode_all(sd, coded, clen, outbuf, n_info + 16);

  int errors = 0;
  for (int i = 0; i < n_info; ++i) errors += (outbuf[i] != msg[i]);
  printf("standard K7R12: decoded %d bits, errors=%d\n", got, errors);
  assert(got == n_info && errors == 0);

  assert(dv_code_create_standard((dv_standard_code)999) == NULL);

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
  dv_code *code = dv_code_create(K, GENERATORS, N_GEN);
  assert(code != NULL);

  const int n_info = 400;
  uint8_t *msg = malloc((size_t)n_info);
  for (int i = 0; i < n_info; ++i) msg[i] = (uint8_t)((i * 13 + 7) & 1);

  int clen = n_info * N_GEN;
  uint8_t *coded = malloc((size_t)clen);
  int st = 0;
  assert(dv_code_encode(code, msg, n_info, &st, coded) == clen);

  /* Start decoding from input step 100 (a group boundary), where the encoder
   * state is some unknown non-zero value. */
  const int splice_step = 100;
  const uint8_t *rx = coded + splice_step * N_GEN;
  int rl = clen - splice_step * N_GEN;
  int avail = n_info - splice_step; /* input bits available */

  dv_stream_decoder *sd = make_decoder(code, 40, 4, 0.01, 0.01, 0.01, 0.0);
  assert(sd != NULL);

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
  printf(
      "blind acquisition: spliced at step %d, decoded %d bits, "
      "post-warmup errors=%d/%d\n",
      splice_step, got, errors, counted);
  assert(counted > 0);
  assert(errors == 0);

  dv_stream_decoder_destroy(sd);
  free(out);
  free(coded);
  free(msg);
  dv_code_destroy(code);
}

/* Minimal settings: leave max_drift (and the indel probabilities) at 0, so the
 * decoder corrects bit flips only. A few scattered flips are still fixed. */
static void test_stream_flips_only(void) {
  dv_code *code = dv_code_create(K, GENERATORS, N_GEN);
  assert(code != NULL);

  const int n_info = 300;
  uint8_t *msg = malloc((size_t)n_info);
  for (int i = 0; i < n_info; ++i) msg[i] = (uint8_t)((i * 7 + 3) & 1);

  int clen = n_info * N_GEN;
  uint8_t *rx = malloc((size_t)clen);
  int st = 0;
  assert(dv_code_encode(code, msg, n_info, &st, rx) == clen);

  int flips[] = {123, 400, 777, 1100, 1450};
  const int nflip = (int)(sizeof(flips) / sizeof(flips[0]));
  for (int i = 0; i < nflip; ++i) rx[flips[i]] ^= 1;

  dv_stream_params params = {
      .decision_depth = 40,
      .p_sub = 0.02,
  };
  dv_stream_decoder *sd = dv_stream_decoder_create(code, &params);
  assert(sd != NULL);

  uint8_t *outbuf = malloc((size_t)n_info + 16);
  int got = stream_decode_all(sd, rx, clen, outbuf, n_info + 16);

  int errors = 0;
  for (int i = 0; i < n_info; ++i) errors += (outbuf[i] != msg[i]);
  printf("stream flips-only: %d flips, decoded %d bits, errors=%d\n", nflip,
         got, errors);
  assert(got == n_info && errors == 0);

  dv_stream_decoder_destroy(sd);
  free(outbuf);
  free(rx);
  free(msg);
  dv_code_destroy(code);
}

/* Every standard preset (the four defaults plus their two alternates each) must
 * create, report the right rate/K, and round-trip a clean stream exactly. */
static void test_all_presets(void) {
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

  for (int idx = 0; idx < np; ++idx) {
    dv_code *code = dv_code_create_standard(presets[idx].code);
    assert(code != NULL);
    assert(dv_code_n(code) == presets[idx].n);
    assert(dv_code_k(code) == presets[idx].k);

    const int n_info = 200;
    uint8_t msg[200];
    for (int i = 0; i < n_info; ++i) msg[i] = (uint8_t)((i * 5 + 1) & 1);
    int clen = n_info * presets[idx].n;
    uint8_t *coded = malloc((size_t)clen);
    int st = 0;
    assert(dv_code_encode(code, msg, n_info, &st, coded) == clen);

    dv_stream_decoder *sd =
        make_decoder(code, 8 * presets[idx].k, 4, 0.01, 0.01, 0.01, 0.0);
    assert(sd != NULL);
    int cap = n_info + 32;
    uint8_t *out = malloc((size_t)cap);
    int got = stream_decode_all(sd, coded, clen, out, cap);
    int cmp = got < n_info ? got : n_info, errors = 0;
    for (int i = 0; i < cmp; ++i) errors += (out[i] != msg[i]);
    assert(got == n_info && errors == 0);

    dv_stream_decoder_destroy(sd);
    free(out);
    free(coded);
    dv_code_destroy(code);
  }
  printf("all presets: %d codes create and round-trip cleanly\n", np);
}

/* The lock-probability output rises to ~1 on a clean coded stream (the decoder
 * is synchronized) and stays low on random, non-coded input (it never locks).
 * A NULL lock pointer is also accepted. */
static void test_lock_probability(void) {
  dv_code *code = dv_code_create(K, GENERATORS, N_GEN);
  assert(code != NULL);

  const int n_info = 600;
  uint8_t *msg = malloc((size_t)n_info);
  for (int i = 0; i < n_info; ++i) msg[i] = (uint8_t)((i * 7 + 3) & 1);

  int clen = n_info * N_GEN;
  uint8_t *coded = malloc((size_t)clen);
  int st = 0;
  assert(dv_code_encode(code, msg, n_info, &st, coded) == clen);

  int cap = n_info + 32;
  uint8_t *out = malloc((size_t)cap);
  double *lock = malloc((size_t)cap * sizeof(double));

  /* Clean coded stream: feed it all at once, capturing per-bit lock prob. */
  dv_stream_decoder *sd = make_decoder(code, 48, 4, 0.01, 0.01, 0.01, 0.0);
  assert(sd != NULL);
  int got = dv_stream_decode(sd, coded, clen, out, lock, cap);
  assert(got > 0);
  double clean_sum = 0.0;
  for (int i = 0; i < got; ++i) {
    assert(lock[i] > 0.0 && lock[i] <= 1.0 + 1e-9); /* it is a probability */
    if (i >= got / 2) clean_sum += lock[i];
  }
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
  assert(sd != NULL);
  int got2 = dv_stream_decode(sd, rnd, clen, out, lock, cap);
  assert(got2 > 0);
  double rnd_sum = 0.0;
  for (int i = got2 / 2; i < got2; ++i) rnd_sum += lock[i];
  double rnd_mean = rnd_sum / (got2 - got2 / 2);
  dv_stream_decoder_destroy(sd);

  printf("lock probability: clean mean=%.3f, random mean=%.3f\n", clean_mean,
         rnd_mean);
  assert(clean_mean > 0.85);              /* locked on real coded data */
  assert(rnd_mean < 0.6);                 /* never locks on noise       */
  assert(clean_mean - rnd_mean > 0.3);    /* clearly discriminates       */

  /* A NULL lock pointer must be accepted and decode normally. */
  sd = make_decoder(code, 48, 4, 0.01, 0.01, 0.01, 0.0);
  assert(sd != NULL);
  assert(dv_stream_decode(sd, coded, clen, out, NULL, cap) > 0);
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
  assert(k3 != NULL && k7 != NULL);

  double k3_k3 = lock_mean_cross(k3, k3);
  double k7_k7 = lock_mean_cross(k7, k7);
  double k3_k7 = lock_mean_cross(k3, k7);
  double k7_k3 = lock_mean_cross(k7, k3);
  printf(
      "lock probability: match k3=%.3f k7=%.3f, cross k3->k7=%.3f k7->k3=%.3f\n",
      k3_k3, k7_k7, k3_k7, k7_k3);

  assert(k3_k3 > 0.85 && k7_k7 > 0.85);          /* matching code locks       */
  assert(k3_k7 < 0.75 && k7_k3 < 0.75);          /* the other code does not    */
  assert(k3_k3 - k3_k7 > 0.3 && k7_k7 - k7_k3 > 0.3);

  dv_code_destroy(k3);
  dv_code_destroy(k7);
}

/* Within each (K, rate) family the default and its two alternates are picked to
 * be mutually distinguishable: each locks onto its own stream but not onto a
 * sibling's. (Across families - different K or rate - this is NOT guaranteed;
 * see the comment on dv_standard_code.) */
static void test_cross_lock_within_family(void) {
  struct {
    const char *name;
    dv_standard_code v[3];
  } fam[] = {
      {"K3_R1_2",
       {DV_CODE_K3_RATE_1_2, DV_CODE_K3_RATE_1_2_ALT1, DV_CODE_K3_RATE_1_2_ALT2}},
      {"K7_R1_2",
       {DV_CODE_K7_RATE_1_2, DV_CODE_K7_RATE_1_2_ALT1, DV_CODE_K7_RATE_1_2_ALT2}},
      {"K7_R1_3",
       {DV_CODE_K7_RATE_1_3, DV_CODE_K7_RATE_1_3_ALT1, DV_CODE_K7_RATE_1_3_ALT2}},
      {"K5_R1_5",
       {DV_CODE_K5_RATE_1_5, DV_CODE_K5_RATE_1_5_ALT1, DV_CODE_K5_RATE_1_5_ALT2}},
  };
  const int nf = (int)(sizeof(fam) / sizeof(fam[0]));

  double max_cross = 0.0, min_self = 1.0;
  for (int f = 0; f < nf; ++f) {
    dv_code *code[3];
    for (int i = 0; i < 3; ++i) {
      code[i] = dv_code_create_standard(fam[f].v[i]);
      assert(code[i] != NULL);
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        double m = lock_mean_cross(code[i], code[j]);
        if (i == j) {
          if (m < min_self) min_self = m;
          assert(m > 0.9); /* a code locks onto its own stream */
        } else {
          if (m > max_cross) max_cross = m;
          assert(m < 0.75); /* but not onto a sibling's */
        }
      }
    }
    for (int i = 0; i < 3; ++i) dv_code_destroy(code[i]);
  }
  printf("cross-lock within family: max sibling=%.3f, min self=%.3f\n",
         max_cross, min_self);
}

static void test_error_paths(void) {
  /* Code creation rejects bad arguments by returning NULL. */
  assert(dv_code_create(1, GENERATORS, N_GEN) == NULL); /* K < 2 */
  assert(dv_code_create(K, GENERATORS, 0) == NULL);     /* n < 1 */
  assert(dv_code_n(NULL) == -1);

  dv_code *code = dv_code_create(K, GENERATORS, N_GEN);
  assert(code != NULL);

  /* Encoder rejects an out-of-range carried-in state. */
  uint8_t bit = DV_TRUE, obuf[N_GEN];
  int badstate = 1 << 20;
  assert(dv_code_encode(code, &bit, 1, &badstate, obuf) == DV_ERR_ARG);

  /* Decoder creation rejects bad settings by returning NULL. */
  dv_stream_params ok = {
      .decision_depth = 40,
      .max_drift = 4,
      .p_sub = 0.01,
      .p_ins = 0.01,
      .p_del = 0.01,
  };
  assert(dv_stream_decoder_create(NULL, &ok) == NULL);  /* null code   */
  assert(dv_stream_decoder_create(code, NULL) == NULL); /* null params */

  dv_stream_params p;
  p = ok;
  p.decision_depth = 0;
  assert(dv_stream_decoder_create(code, &p) == NULL);
  p = ok;
  p.max_drift = -1;
  assert(dv_stream_decoder_create(code, &p) == NULL);
  p = ok;
  p.p_sub = 0.0;
  assert(dv_stream_decoder_create(code, &p) == NULL);
  p = ok;
  p.p_ins = p.p_del = 0.6;
  assert(dv_stream_decoder_create(code, &p) == NULL);
  p = ok;
  p.p_erase = 1.0;
  assert(dv_stream_decoder_create(code, &p) == NULL);
  /* max_drift > 0 needs insertion/deletion probabilities. */
  p = ok;
  p.p_ins = p.p_del = 0.0;
  assert(dv_stream_decoder_create(code, &p) == NULL);

  dv_stream_decoder *sd = dv_stream_decoder_create(code, &ok);
  assert(sd != NULL);

  /* Streaming decode argument checks. */
  uint8_t out8 = 0;
  assert(dv_stream_decode(NULL, &bit, 1, &out8, NULL, 1) == DV_ERR_ARG);
  assert(dv_stream_decode(sd, NULL, 1, &out8, NULL, 1) == DV_ERR_ARG); /* null in */
  assert(dv_stream_decode_flush(NULL, &out8, 1) == DV_ERR_ARG);

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
  printf("all tests passed\n");
  return 0;
}
