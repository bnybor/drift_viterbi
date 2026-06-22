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
 * Tests for the multi-decoder: running one received stream through a family of
 * same-rate decoders and, per bit, keeping the value from whichever is most
 * confidently locked (else DV_ERASURE). Covers locking onto the true code and
 * reporting its index, recovery under each channel impairment - flips, erasures,
 * insertions, deletions - and all of them combined, abstaining on random data,
 * the lock thresholds, and argument handling. The data-driven tests are
 * Monte-Carlo loops fanned across cores with OpenMP (a no-op without it).
 */

#include "dv_test_util.h"

/* Three mutually distinguishable rate-1/3, K=7 codes: same n and constraint
 * length, different generators - exactly the lockstep family the multi-decoder
 * expects. */
static const dv_standard_code FAMILY[] = {
    DV_CODE_K7_RATE_1_3,
    DV_CODE_K7_RATE_1_3_ALT1,
    DV_CODE_K7_RATE_1_3_ALT2,
};
#define N_DEC ((int)(sizeof(FAMILY) / sizeof(FAMILY[0])))
#define DEPTH 48

/* Build a multi-decoder over the whole family with identical decoder settings,
 * handing back the codes (which must outlive the multi-decoder) via `codes`. */
static dv_multi_decoder *build_multi(dv_code *codes[N_DEC],
                                     const dv_multi_params *params_in) {
  for (int j = 0; j < N_DEC; ++j) {
    codes[j] = dv_code_create_standard(FAMILY[j]);
    assert(codes[j]);
  }
  dv_multi_params params = params_in ? *params_in : (dv_multi_params){0};
  params.codes = (const dv_code *const *)codes;
  params.codes_len = N_DEC;
  if (params.stream.decision_depth == 0) { /* caller left stream unset */
    params.stream = (dv_stream_params){.decision_depth = DEPTH, .p_sub = 0.02};
  }
  return dv_multi_create(&params);
}

static void destroy_codes(dv_code *codes[N_DEC]) {
  for (int j = 0; j < N_DEC; ++j) {
    dv_code_destroy(codes[j]);
  }
}

/* Push a received buffer through the multi-decoder in small chunks, recording
 * the chosen decoder per bit, then drain. Returns the decoded-bit count. */
static int multi_decode_all(dv_multi_decoder *md, const uint8_t *rx, int rl,
                            uint8_t *out, int *locked, int cap) {
  for (int i = 0; i < cap; ++i) {
    locked[i] = -1; /* flush region reports no winner; default to none */
  }
  int got = 0;
  for (int pos = 0; pos < rl;) {
    int chunk = (rl - pos < 41) ? (rl - pos) : 41;
    int w = dv_multi_decode(md, rx + pos, chunk, out + got, locked + got,
                            cap - got);
    assert(w >= 0);
    got += w;
    pos += chunk;
  }
  for (;;) {
    int w = dv_multi_decode_flush(md, out + got, cap - got);
    assert(w >= 0);
    if (w == 0) break;
    got += w;
  }
  return got;
}

#define N_INFO 600     /* message bits per trial                          */
#define MAX_DRIFT 8     /* drift budget for the indel tests                */
#define TRIALS 4        /* Monte-Carlo trials per test                     */
#define TAIL_GUARD 48   /* end-of-stream bits excluded from the bit check  */

/* Decoder settings that can ride out every impairment: flips, erasures, and
 * cumulative insertion/deletion drift (re-anchoring past max_drift). */
static const dv_stream_params IMPAIRED_STREAM = {.decision_depth = DEPTH,
                                                 .max_drift = MAX_DRIFT,
                                                 .p_sub = 0.02,
                                                 .p_ins = 0.01,
                                                 .p_del = 0.01,
                                                 .p_erase = 0.05};

/* One clean trial: a fresh random message encoded with code `true_idx`, fed
 * verbatim (no channel). The multi-decoder must recover it exactly, attribute
 * every decoded bit to true_idx, and decode the end-of-stream tail. */
static void clean_trial(uint64_t seed, int true_idx) {
  uint64_t rng = seed;
  dv_code *codes[N_DEC];
  dv_multi_decoder *md = build_multi(codes, NULL); /* default thresholds */
  assert(md != NULL);

  uint8_t *msg = malloc((size_t)N_INFO);
  rand_bits(msg, N_INFO, &rng);

  int clen = N_INFO * dv_code_n(codes[true_idx]);
  uint8_t *coded = malloc((size_t)clen);
  int st = 0;
  dv_code_encode(codes[true_idx], msg, N_INFO, &st, coded);

  int cap = N_INFO + 64;
  uint8_t *out = malloc((size_t)cap);
  int *locked = malloc((size_t)cap * sizeof(int));
  int got = multi_decode_all(md, coded, clen, out, locked, cap);

  int decoded = 0, bit_errors = 0, wrong_idx = 0;
  for (int i = 0; i < got; ++i) {
    if (out[i] == DV_ERASURE) continue;
    ++decoded;
    if (i < N_INFO && out[i] != msg[i]) ++bit_errors;
    /* The flush tail reports no index (flush has no locked_decoder out-param),
     * so check attribution only where one is set. */
    if (locked[i] >= 0 && locked[i] != true_idx) ++wrong_idx;
  }
  check("clean: output aligns 1:1 with message", got == N_INFO);
  check("clean: no errors among decoded bits", bit_errors == 0);
  check("clean: decoded bits attributed to true code", wrong_idx == 0);
  check_gt("clean: most bits decoded", (double)decoded, (double)N_INFO / 2);
  check("clean: final bit decoded, not erased",
        out[N_INFO - 1] != DV_ERASURE && out[N_INFO - 1] == msg[N_INFO - 1]);

  free(locked);
  free(out);
  free(coded);
  free(msg);
  dv_multi_destroy(md);
  destroy_codes(codes);
}

/* One impairment trial: a fresh random message encoded with code `true_idx`,
 * pushed through deletions -> insertions -> flips -> erasures at the given rates
 * (any may be 0), then decoded with IMPAIRED_STREAM. The multi-decoder must lock
 * onto true_idx (no bit attributed to a wrong code), recover the message on its
 * non-erased in-range bits (within `max_errs`), and decode at least `min_frac`
 * of the message. `label` tags the printed line. */
static void recover_trial(uint64_t seed, int true_idx, double p_sub,
                          double p_ins, double p_del, double p_erase,
                          double min_frac, int max_errs, const char *label) {
  uint64_t rng = seed;
  dv_code *codes[N_DEC];
  dv_multi_params params = {.stream = IMPAIRED_STREAM};
  dv_multi_decoder *md = build_multi(codes, &params);
  assert(md != NULL);

  uint8_t *msg = malloc((size_t)N_INFO);
  rand_bits(msg, N_INFO, &rng);

  size_t clen = (size_t)N_INFO * dv_code_n(codes[true_idx]);
  uint8_t *coded = malloc(clen);
  int st = 0;
  dv_code_encode(codes[true_idx], msg, N_INFO, &st, coded);

  /* Channel chain: indels first (they reframe), then flips, then erasures. */
  uint8_t *del = malloc(clen);
  size_t ldel = delete_channel(coded, clen, p_del, &rng, del);
  uint8_t *rx = malloc(2 * clen + 64);
  size_t rl = insert_channel(del, ldel, p_ins, &rng, rx);
  flip_channel(rx, rl, p_sub, &rng);
  erase_channel(rx, rl, p_erase, &rng);

  int cap = N_INFO + 64;
  uint8_t *out = malloc((size_t)cap);
  int *locked = malloc((size_t)cap * sizeof(int));
  int got = multi_decode_all(md, rx, (int)rl, out, locked, cap);

  /* Compare over the settled region only: a deletion or insertion in the final
   * bits can be left unresolved at the flush boundary (a +/-1 shift of the very
   * tail), exactly as leading bits are unreliable during acquisition. The
   * settled-region check still catches any real mid-stream desync. */
  int cmp = got < N_INFO ? got : N_INFO;
  int settled = cmp > TAIL_GUARD ? cmp - TAIL_GUARD : 0;
  int decoded = 0, bit_errors = 0, wrong_idx = 0;
  for (int i = 0; i < got; ++i) {
    if (out[i] == DV_ERASURE) continue;
    ++decoded;
    if (i < settled && out[i] != msg[i]) ++bit_errors;
    if (locked[i] >= 0 && locked[i] != true_idx) ++wrong_idx;
  }
  printf("  [%s] got=%d decoded=%d settled_errors=%d wrong_idx=%d\n", label, got,
         decoded, bit_errors, wrong_idx);
  check("recovered settled bits match message", bit_errors <= max_errs);
  check("no bits attributed to wrong code", wrong_idx == 0);
  check_gt("locked onto true code", (double)decoded, min_frac * N_INFO);
  /* The decoder resolves nearly all the drift: output length stays close to the
   * message length (a small residue at the unresolved tail is allowed). */
  check("output length tracks message", got >= N_INFO - TAIL_GUARD &&
                                            got <= N_INFO + TAIL_GUARD);

  free(locked);
  free(out);
  free(rx);
  free(del);
  free(coded);
  free(msg);
  dv_multi_destroy(md);
  destroy_codes(codes);
}

/* One unit of work for the suite's data-driven tests. All such trials are
 * independent (own PRNG, own buffers), so they are gathered into a single flat
 * list and run in one wide parallel region rather than many narrow ones. */
typedef enum { TK_CLEAN, TK_RECOVER, TK_NOISE, TK_MARGIN } trial_kind;
typedef struct {
  trial_kind kind;
  uint64_t seed;
  int true_idx;
  double p_sub, p_ins, p_del, p_erase, min_frac;
  int max_errs;
  const char *label;
} trial_desc;

/* Random data fits no code, so the multi-decoder should abstain: almost every
 * bit erased, none confidently attributed to a decoder. */
static void noise_trial(uint64_t seed) {
  uint64_t rng = seed;
  dv_code *codes[N_DEC];
  dv_multi_decoder *md = build_multi(codes, &(dv_multi_params){0});
  assert(md != NULL);

  const int rl = 1800;
  uint8_t *rx = malloc((size_t)rl);
  rand_bits(rx, rl, &rng);

  int cap = rl + 64;
  uint8_t *out = malloc((size_t)cap);
  int *locked = malloc((size_t)cap * sizeof(int));
  int got = multi_decode_all(md, rx, rl, out, locked, cap);

  int decoded = 0;
  for (int i = 0; i < got; ++i) {
    if (out[i] != DV_ERASURE) ++decoded;
  }
  /* A spurious lock here and there is tolerable; a confident decode is not. */
  check_lt("random data mostly erased", (double)decoded, (double)got * 0.1);

  free(locked);
  free(out);
  free(rx);
  dv_multi_destroy(md);
  destroy_codes(codes);
}

/* dv_multi_params takes effect: an unsatisfiable lock_margin (no two lock
 * probabilities in [0,1] can differ by more than 1) makes the selector abstain
 * on every bit, so a stream the defaults decode in full is now fully erased. */
static void margin_trial(uint64_t seed) {
  uint64_t rng = seed;
  dv_multi_params params = {.lock_margin = 1.5};
  dv_code *codes[N_DEC];
  dv_multi_decoder *md = build_multi(codes, &params);
  assert(md != NULL);

  uint8_t *msg = malloc((size_t)N_INFO);
  rand_bits(msg, N_INFO, &rng);
  int clen = N_INFO * dv_code_n(codes[0]);
  uint8_t *coded = malloc((size_t)clen);
  int st = 0;
  dv_code_encode(codes[0], msg, N_INFO, &st, coded);

  int cap = N_INFO + 64;
  uint8_t *out = malloc((size_t)cap);
  int *locked = malloc((size_t)cap * sizeof(int));
  int got = multi_decode_all(md, coded, clen, out, locked, cap);

  int decoded = 0;
  for (int i = 0; i < got; ++i) {
    if (out[i] != DV_ERASURE) ++decoded;
  }
  check("unsatisfiable lock_margin erases every bit", decoded == 0);

  free(locked);
  free(out);
  free(coded);
  free(msg);
  dv_multi_destroy(md);
  destroy_codes(codes);
}

/* Dispatch one descriptor to the matching trial helper. */
static void run_trial(const trial_desc *d) {
  switch (d->kind) {
    case TK_CLEAN:
      clean_trial(d->seed, d->true_idx);
      break;
    case TK_RECOVER:
      recover_trial(d->seed, d->true_idx, d->p_sub, d->p_ins, d->p_del,
                    d->p_erase, d->min_frac, d->max_errs, d->label);
      break;
    case TK_NOISE:
      noise_trial(d->seed);
      break;
    case TK_MARGIN:
      margin_trial(d->seed);
      break;
  }
}

/* Argument handling: NULL decoder, negative sizes, and the empty-decoder edge. */
static void test_multi_args(void) {
  printf("test_multi_args\n");
  uint8_t in[8] = {0}, out[8] = {0};
  check("null decoder rejected",
        dv_multi_decode(NULL, in, 8, out, NULL, 8) == DV_ERR_ARG);
  check("null flush decoder rejected",
        dv_multi_decode_flush(NULL, out, 8) == DV_ERR_ARG);

  check("null params rejected", dv_multi_create(NULL) == NULL);

  dv_multi_params empty_params = {0}; /* no decoders */
  dv_multi_decoder *empty = dv_multi_create(&empty_params);
  REQUIRE("empty multi-decoder created", empty != NULL);
  check("negative n_in rejected",
        dv_multi_decode(empty, in, -1, out, NULL, 8) == DV_ERR_ARG);
  check("empty decoder yields no output",
        dv_multi_decode(empty, in, 8, out, NULL, 8) == 0);
  check("empty decoder flush yields no output",
        dv_multi_decode_flush(empty, out, 8) == 0);
  dv_multi_destroy(empty);
}

/* Two decoders for the SAME code lock equally well, so a hard "best minus
 * next-best lock" selector would see ~zero lead and erase every bit. Likelihood-
 * weighted combining instead lets their agreeing votes reinforce, so the stream
 * decodes in full - the behaviour that distinguishes soft combining from picking
 * a single winner. */
static void duplicate_code_trial(uint64_t seed) {
  uint64_t rng = seed;
  dv_code *a = dv_code_create_standard(DV_CODE_K7_RATE_1_3);
  dv_code *b = dv_code_create_standard(DV_CODE_K7_RATE_1_3);
  assert(a && b);
  const dv_code *codes[] = {a, b};
  dv_multi_decoder *md = dv_multi_create(&(dv_multi_params){
      .codes = codes,
      .codes_len = 2,
      .stream = {.decision_depth = DEPTH, .p_sub = 0.02}});
  assert(md != NULL);

  uint8_t *msg = malloc((size_t)N_INFO);
  rand_bits(msg, N_INFO, &rng);
  int clen = N_INFO * dv_code_n(a);
  uint8_t *coded = malloc((size_t)clen);
  int st = 0;
  dv_code_encode(a, msg, N_INFO, &st, coded);

  int cap = N_INFO + 64;
  uint8_t *out = malloc((size_t)cap);
  int *locked = malloc((size_t)cap * sizeof(int));
  int got = multi_decode_all(md, coded, clen, out, locked, cap);

  int decoded = 0, bit_errors = 0;
  for (int i = 0; i < got; ++i) {
    if (out[i] == DV_ERASURE) continue;
    ++decoded;
    if (i < N_INFO && out[i] != msg[i]) ++bit_errors;
  }
  check("duplicate-code: output aligns 1:1 with message", got == N_INFO);
  check("duplicate-code: no errors among decoded bits", bit_errors == 0);
  check_gt("duplicate-code: agreeing votes commit rather than erase",
           (double)decoded, (double)N_INFO / 2);

  free(locked);
  free(out);
  free(coded);
  free(msg);
  dv_multi_destroy(md);
  dv_code_destroy(a);
  dv_code_destroy(b);
}

/* A code set that mixes constraint lengths (so different n_states) shares a rate
 * but not a trellis geometry; dv_multi_create must reject it rather than let a
 * trellis sized for the larger code index the smaller code's tables out of
 * bounds. */
static void test_reject_incompatible(void) {
  dv_code *k7 = dv_code_create_standard(DV_CODE_K7_RATE_1_2);
  dv_code *k3 = dv_code_create_standard(DV_CODE_K3_RATE_1_2);
  assert(k7 && k3);
  const dv_code *mixed[] = {k7, k3}; /* same dv_code_n (2), different dv_code_k */
  dv_multi_decoder *md = dv_multi_create(&(dv_multi_params){
      .codes = mixed,
      .codes_len = 2,
      .stream = {.decision_depth = DEPTH, .p_sub = 0.02}});
  check("multi rejects mixed constraint lengths", md == NULL);
  dv_multi_destroy(md); /* NULL-safe */
  dv_code_destroy(k7);
  dv_code_destroy(k3);
}

int main(void) {
  const uint64_t seed = 0xD1F7C0DEULL;

  /* Gather every data-driven trial into one flat list so a single wide parallel
   * region keeps all cores busy, instead of eight narrow regions (TRIALS wide
   * each) with a barrier between them. Seeds and parameters are exactly as when
   * these were separate per-impairment tests, so results are unchanged. */
  trial_desc trials[8 * TRIALS];
  int n = 0;
  /* Heaviest trials (the drift-tracking recover kinds, max_drift wide) first, so
   * schedule(dynamic) packs them early and the tail is the light clean/noise/
   * margin trials - a longest-processing-time ordering that shortens makespan. */
  for (int t = 0; t < TRIALS; ++t) {
    /* flips: substitution noise only (no drift) */
    trials[n++] = (trial_desc){.kind = TK_RECOVER,
                               .seed = seed + 100 + (uint64_t)t,
                               .true_idx = t % N_DEC,
                               .p_sub = 0.02,
                               .min_frac = 0.6,
                               .label = "flips"};
    /* erasures only */
    trials[n++] = (trial_desc){.kind = TK_RECOVER,
                               .seed = seed + 200 + (uint64_t)t,
                               .true_idx = t % N_DEC,
                               .p_erase = 0.08,
                               .min_frac = 0.6,
                               .label = "erasures"};
    /* insertions only: cumulative positive drift past max_drift */
    trials[n++] = (trial_desc){.kind = TK_RECOVER,
                               .seed = seed + 300 + (uint64_t)t,
                               .true_idx = t % N_DEC,
                               .p_ins = 0.01,
                               .min_frac = 0.4,
                               .label = "insertions"};
    /* deletions only: cumulative negative drift */
    trials[n++] = (trial_desc){.kind = TK_RECOVER,
                               .seed = seed + 400 + (uint64_t)t,
                               .true_idx = t % N_DEC,
                               .p_del = 0.01,
                               .min_frac = 0.4,
                               .label = "deletions"};
    /* combined: all impairments at modest, trackable rates */
    trials[n++] = (trial_desc){.kind = TK_RECOVER,
                               .seed = seed + 500 + (uint64_t)t,
                               .true_idx = t % N_DEC,
                               .p_sub = 0.008,
                               .p_ins = 0.003,
                               .p_del = 0.003,
                               .p_erase = 0.02,
                               .min_frac = 0.25,
                               .label = "combined"};
  }
  /* Light trials (no drift tracking) last. */
  for (int t = 0; t < TRIALS; ++t) {
    /* clean: cycle the true code across the family to exercise index reporting */
    trials[n++] = (trial_desc){
        .kind = TK_CLEAN, .seed = seed + (uint64_t)t, .true_idx = t % N_DEC};
    trials[n++] = (trial_desc){.kind = TK_NOISE, .seed = seed + 600 + (uint64_t)t};
    trials[n++] =
        (trial_desc){.kind = TK_MARGIN, .seed = seed + 700 + (uint64_t)t};
  }

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < n; ++i) {
    run_trial(&trials[i]);
  }

  test_multi_args(); /* serial: uses REQUIRE, must not run inside a trial */
  duplicate_code_trial(seed + 800); /* soft combining: agreeing votes reinforce */
  test_reject_incompatible();       /* incompatible code sets are refused       */
  return test_summary("multi");
}
