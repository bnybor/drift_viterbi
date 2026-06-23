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
 * dv_metrics - Monte-Carlo measurement of decoding-mistake rate as a function
 * of the channel's flip / insert / delete / erase rates, for each standard code.
 *
 * For one data point we: generate a random message, encode it (with a flush so
 * the message bits sit in the stream interior), pass the coded bits through a
 * channel that independently flips, inserts, deletes, and erases (marks lost)
 * bits, stream-decode, and count how many decoded bits disagree with the
 * original message.
 *
 * The reported metric is the normalized EDIT (Levenshtein) distance between the
 * decoded bits and the original message, divided by the number of message bits.
 * Edit distance counts the substitutions, insertions, and deletions needed to
 * turn one into the other, so a single uncorrected sync slip costs one edit
 * rather than misaligning (and thus mis-scoring) the whole remaining stream -
 * the right metric when the channel itself inserts and deletes bits. A first
 * `warmup` bits are dropped from both sequences to skip the decoder's
 * blind-acquisition transient, and the trailing flush bits are trimmed off.
 *
 * Each axis (flip, insert, delete) is swept independently with the other two
 * rates held at zero, so each curve isolates one channel impairment.
 *
 * Alongside the edit rate we report two confidence metrics, both in [0, 1]:
 *
 *   mean_lock   - the decoder's own running estimate (see dv_stream_decode's
 *                 lock_probability) that it is tracking a valid coded stream,
 *                 averaged across the kept bits. It shows how confidently the
 *                 decoder stays synced as each impairment ramps up.
 *   mean_detect - the blind detector's confidence (dv_detect) that the stream
 *                 carries *any* rate-1/n, constraint-length-k code, with no
 *                 knowledge of the generators or the sent bits, averaged over a
 *                 sliding window across the received bits. It shows how
 *                 recognizable the coded structure stays as the channel ramps.
 *
 * Output is CSV on stdout (see header row); feed it to metrics/plot_metrics.py.
 *
 * Usage: dv_metrics [trials] [info_bits] [seed]
 */

#include <drift_viterbi/drift_viterbi.h>

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* -- deterministic PRNG (splitmix64) --------------------------------------- */

static uint64_t rng_next(uint64_t *state) {
  uint64_t value = (*state += 0x9E3779B97F4A7C15ULL);
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31);
}

/* Uniform double in [0, 1). */
static double rng_unit(uint64_t *state) {
  return (double)(rng_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

/* Independent, reproducible seed for work item `index` from the base seed. Each
 * point owns its own PRNG stream, so results don't depend on thread scheduling
 * or how many draws other points made - parallel runs match serial ones. */
static uint64_t derive_seed(uint64_t base, int index) {
  uint64_t value = base + 0x9E3779B97F4A7C15ULL * (uint64_t)(index + 1);
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31);
}

/* -- helpers --------------------------------------------------------------- */

static void *xmalloc(size_t size) {
  void *ptr = malloc(size);
  if (!ptr) {
    fprintf(stderr, "dv_metrics: out of memory\n");
    exit(1);
  }
  return ptr;
}

/* Push a received buffer through the streaming decoder in small chunks, then
 * drain. Returns the number of decoded bits collected (<= decoded_cap). When
 * `lock` is non-NULL it receives the per-bit lock probability for the bits
 * emitted by streaming; the trailing flush bits carry no lock value, so
 * *n_stream (if non-NULL) reports how many leading bits `lock` was filled for. */
static int decode_all(dv_stream_decoder *decoder, const uint8_t *received,
                      int received_len, uint8_t *decoded, double *lock,
                      int decoded_cap, int *n_stream) {
  int n_decoded = 0, read_pos = 0;
  while (read_pos < received_len && n_decoded < decoded_cap) {
    int chunk = received_len - read_pos < 64 ? received_len - read_pos : 64;
    int written = dv_stream_decode(decoder, received + read_pos, chunk,
                                   decoded + n_decoded,
                                   lock ? lock + n_decoded : NULL,
                                   decoded_cap - n_decoded);
    if (written < 0) {
      return written;
    }
    n_decoded += written;
    read_pos += chunk;
  }
  if (n_stream) {
    *n_stream = n_decoded;
  }
  for (;;) {
    if (n_decoded >= decoded_cap) {
      break;
    }
    int written = dv_stream_decode_flush(decoder, decoded + n_decoded,
                                         decoded_cap - n_decoded);
    if (written < 0) {
      return written;
    }
    if (written == 0) {
      break;
    }
    n_decoded += written;
  }
  return n_decoded;
}

/* Apply the channel to `coded` (coded_len bits): with probability p_ins emit a
 * random bit before each coded bit, with probability p_del drop the coded bit,
 * otherwise emit it flipped with probability p_sub and then, with probability
 * p_erase, marked DV_ERASURE (received but known-lost). Returns the received
 * length and stores a freshly malloc'd buffer in *out_received. */
static int apply_channel(const uint8_t *coded, int coded_len, double p_sub,
                         double p_ins, double p_del, double p_erase,
                         uint64_t *rng, uint8_t **out_received) {
  int capacity = coded_len + coded_len / 4 + 64;
  uint8_t *received = xmalloc((size_t)capacity);
  int received_len = 0;
  for (int i = 0; i < coded_len; ++i) {
    if (received_len + 2 > capacity) {
      capacity *= 2;
      uint8_t *grown = realloc(received, (size_t)capacity);
      if (!grown) {
        free(received);
        fprintf(stderr, "dv_metrics: out of memory\n");
        exit(1);
      }
      received = grown;
    }
    if (p_ins > 0.0 && rng_unit(rng) < p_ins) {
      received[received_len++] = (uint8_t)(rng_next(rng) & 1u);
    }
    if (p_del > 0.0 && rng_unit(rng) < p_del) {
      continue;
    }
    uint8_t bit = coded[i];
    if (p_sub > 0.0 && rng_unit(rng) < p_sub) {
      bit ^= 1u;
    }
    if (p_erase > 0.0 && rng_unit(rng) < p_erase) {
      bit = DV_ERASURE;
    }
    received[received_len++] = bit;
  }
  *out_received = received;
  return received_len;
}

/* -- edit distance --------------------------------------------------------- */

/* Banded edit distance: returns the true distance if it is <= band, otherwise
 * band + 1. Only the diagonal band |row - col| <= band is computed, so the cost
 * is O((len_a + 1) * band). `prev` and `cur` are caller-owned scratch of length
 * len_b + 1. */
static int edit_threshold(const uint8_t *seq_a, int len_a, const uint8_t *seq_b,
                          int len_b, int band, int *prev, int *cur) {
  if (len_a - len_b > band || len_b - len_a > band) {
    return band + 1; /* lengths alone already differ by more than band edits */
  }
  const int INF = INT_MAX / 4;

  /* Row 0: prefix of seq_b reached by insertions, but only inside the band; the
   * cell just past it (read as the next row's right boundary) is INF. */
  int init_end = band + 1 < len_b ? band + 1 : len_b;
  for (int col = 0; col <= init_end; ++col) {
    prev[col] = col <= band ? col : INF;
  }

  for (int row = 1; row <= len_a; ++row) {
    int col_lo = row - band > 1 ? row - band : 1;
    int col_hi = row + band < len_b ? row + band : len_b;
    int right_bound = col_hi + 1 < len_b ? col_hi + 1 : len_b;
    for (int col = col_lo - 1; col <= right_bound; ++col) {
      cur[col] = INF; /* clear only the band cells we will read or write */
    }
    if (row <= band) {
      cur[0] = row; /* delete the first `row` chars of seq_a */
    }
    for (int col = col_lo; col <= col_hi; ++col) {
      int subst_cost = prev[col - 1] + (seq_a[row - 1] != seq_b[col - 1] ? 1 : 0);
      int del_cost = prev[col] + 1;
      int ins_cost = cur[col - 1] + 1;
      int best = subst_cost < del_cost ? subst_cost : del_cost;
      cur[col] = ins_cost < best ? ins_cost : best;
    }
    int *tmp = prev;
    prev = cur;
    cur = tmp;
  }

  int dist = prev[len_b];
  return dist > band ? band + 1 : dist;
}

/* Edit distance between seq_a and seq_b, computed with an exponentially growing
 * band so that close sequences (the common case) cost only O(distance *
 * length). The true distance never exceeds max(len_a, len_b), which bounds the
 * worst case. */
static long edit_distance(const uint8_t *seq_a, int len_a, const uint8_t *seq_b,
                          int len_b, int *prev, int *cur) {
  const int max_dist = len_a > len_b ? len_a : len_b;
  for (int band = 8;; band *= 2) {
    if (band > max_dist) {
      band = max_dist;
    }
    int dist = edit_threshold(seq_a, len_a, seq_b, len_b, band, prev, cur);
    if (dist <= band || band >= max_dist) {
      return dist;
    }
  }
}

/* -- experiment ------------------------------------------------------------ */

typedef struct {
  const char *name;
  dv_standard_code which;
} code_entry;

static const code_entry CODES[] = {
    {"K3_R1_2", DV_CODE_K3_RATE_1_2},
    {"K7_R1_2", DV_CODE_K7_RATE_1_2},
    {"K7_R1_3", DV_CODE_K7_RATE_1_3},
    {"K5_R1_5", DV_CODE_K5_RATE_1_5},
};
#define N_CODES ((int)(sizeof(CODES) / sizeof(CODES[0])))

typedef enum { AXIS_FLIP, AXIS_INSERT, AXIS_DELETE, AXIS_ERASE } axis;
static const char *AXIS_NAME[] = {"flip", "insert", "delete", "erase"};
#define N_AXES (AXIS_ERASE + 1)

/* The metrics measured at each point. Run length is derived from the edit rate,
 * so it shares edit's grid and is not a separate entry here. */
typedef enum { METRIC_EDIT, METRIC_LOCK, METRIC_DETECT } metric;
static const char *METRIC_NAME[] = {"edit", "lock", "detect"};
#define N_METRICS ((int)(sizeof(METRIC_NAME) / sizeof(METRIC_NAME[0])))

/* Channel rate grids, one per (metric, impairment) pair. Each is sampled only
 * over the range where the metric carries information on that axis, and within
 * that range the points are clustered where the curve bends - where its slope is
 * changing fastest (the knee onset, the steepest stretch, the rollover) - and
 * thinned across the near-straight runs and flat plateaus. The bend locations
 * below were read off the curvature of the coarse sweep, so the grids differ by
 * both metric and axis. Each grid then subdivides its intervals threefold for
 * resolution, which keeps that bend-weighted spacing (dense stays dense).
 *
 * EDIT (and run length, derived from it): the error-correction knee. Points bunch
 * around the onset and the steep middle (~0.02-0.12 for flip/insert/delete) and
 * stretch out over the gentler tail to ~0.25 where the strongest code saturates.
 * Erasures bend much later, so that grid skips the flat floor below ~0.2 and
 * concentrates on the 0.35-0.92 knee. */
static const double EDIT_FLIP_RATES[] = {
    0, 0.0033, 0.0067, 0.01, 0.0133, 0.0167, 0.02, 0.025, 0.03, 0.035,
    0.04, 0.045, 0.05, 0.0567, 0.0633, 0.07, 0.0767, 0.0833, 0.09, 0.1,
    0.11, 0.12, 0.1333, 0.1467, 0.16, 0.1733, 0.1867, 0.2, 0.2167, 0.2333,
    0.25};
static const double EDIT_INSERT_RATES[] = {
    0, 0.0017, 0.0033, 0.005, 0.0067, 0.0083, 0.01, 0.0133, 0.0167, 0.02,
    0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.0567, 0.0633, 0.07, 0.08,
    0.09, 0.1, 0.1133, 0.1267, 0.14, 0.16, 0.18, 0.2, 0.2167, 0.2333, 0.25};
static const double EDIT_DELETE_RATES[] = {
    0, 0.0033, 0.0067, 0.01, 0.0133, 0.0167, 0.02, 0.0233, 0.0267, 0.03,
    0.0367, 0.0433, 0.05, 0.0567, 0.0633, 0.07, 0.0767, 0.0833, 0.09, 0.1,
    0.11, 0.12, 0.1333, 0.1467, 0.16, 0.1733, 0.1867, 0.2, 0.2167, 0.2333,
    0.25};
static const double EDIT_ERASE_RATES[] = {
    0, 0.0667, 0.1333, 0.2, 0.25, 0.3, 0.35, 0.3833, 0.4167, 0.45, 0.4833,
    0.5167, 0.55, 0.5733, 0.5967, 0.62, 0.6467, 0.6733, 0.7, 0.7267,
    0.7533, 0.78, 0.8033, 0.8267, 0.85, 0.8733, 0.8967, 0.92};

/* LOCK: confidence falls from ~1 to a plateau over the descent (the bends live at
 * the onset and through ~0.02-0.2), then runs flat. Points bunch on the descent
 * and thin across the plateau out to 1.0. Delete is the exception - it bends
 * hard again at 0.9-1.0 as the stream empties out, so points re-cluster there.
 * Erase confidence collapses to zero by ~0.4, with its steepest bend near 0.2. */
static const double LOCK_FLIP_RATES[] = {
    0, 0.0033, 0.0067, 0.01, 0.0133, 0.0167, 0.02, 0.025, 0.03, 0.035,
    0.04, 0.045, 0.05, 0.0567, 0.0633, 0.07, 0.0767, 0.0833, 0.09, 0.1,
    0.11, 0.12, 0.1333, 0.1467, 0.16, 0.1733, 0.1867, 0.2, 0.2167, 0.2333,
    0.25, 0.2833, 0.3167, 0.35, 0.4167, 0.4833, 0.55, 0.6333, 0.7167, 0.8,
    0.8667, 0.9333, 1};
static const double LOCK_INSERT_RATES[] = {
    0, 0.0033, 0.0067, 0.01, 0.0133, 0.0167, 0.02, 0.025, 0.03, 0.035,
    0.04, 0.045, 0.05, 0.0567, 0.0633, 0.07, 0.0767, 0.0833, 0.09, 0.1,
    0.11, 0.12, 0.1333, 0.1467, 0.16, 0.1733, 0.1867, 0.2, 0.22, 0.24,
    0.26, 0.29, 0.32, 0.35, 0.4167, 0.4833, 0.55, 0.6333, 0.7167, 0.8,
    0.8667, 0.9333, 1};
static const double LOCK_DELETE_RATES[] = {
    0, 0.0033, 0.0067, 0.01, 0.0133, 0.0167, 0.02, 0.025, 0.03, 0.035,
    0.04, 0.045, 0.05, 0.0567, 0.0633, 0.07, 0.0767, 0.0833, 0.09, 0.1,
    0.11, 0.12, 0.1333, 0.1467, 0.16, 0.1733, 0.1867, 0.2, 0.2333, 0.2667,
    0.3, 0.3667, 0.4333, 0.5, 0.6, 0.7, 0.8, 0.8333, 0.8667, 0.9, 0.9167,
    0.9333, 0.95, 0.9667, 0.9833, 1};
static const double LOCK_ERASE_RATES[] = {
    0, 0.0067, 0.0133, 0.02, 0.0267, 0.0333, 0.04, 0.05, 0.06, 0.07, 0.08,
    0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2,
    0.21, 0.22, 0.2333, 0.2467, 0.26, 0.2733, 0.2867, 0.3, 0.32, 0.34,
    0.36, 0.4067, 0.4533, 0.5, 0.6667, 0.8333, 1};

/* DETECT: blind detection falls from 1 to 0, but how far that reaches depends on
 * the code's relation window n*(k+1) - the short-window codes (K3_R1_2 most of
 * all) stay detectable far longer, so the flip/insert/delete grids run well past
 * the long-window collapse to follow the short-window descent (to ~0.25-0.30).
 * Points pack into the early collapse and along that descent. Erasures share that
 * window ordering and just collapse monotonically: the long-window codes are gone
 * below ~0.06 while K3_R1_2 holds out to ~0.42, all settling to zero with only a
 * small residual blip near full erasure - so the grid is dense through the early
 * collapse and K3's descent (to ~0.45), with a light flat tail to 0.92. */
static const double DETECT_FLIP_RATES[] = {
    0, 0.0017, 0.0033, 0.005, 0.0067, 0.0083, 0.01, 0.0117, 0.0133, 0.015,
    0.0167, 0.0183, 0.02, 0.0233, 0.0267, 0.03, 0.0333, 0.0367, 0.04,
    0.0467, 0.0533, 0.06, 0.07, 0.08, 0.09, 0.1033, 0.1167, 0.13, 0.1467,
    0.1633, 0.18, 0.2033, 0.2267, 0.25};
static const double DETECT_INSERT_RATES[] = {
    0, 0.0017, 0.0033, 0.005, 0.0067, 0.0083, 0.01, 0.0133, 0.0167, 0.02,
    0.0233, 0.0267, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.0667,
    0.0733, 0.08, 0.09, 0.1, 0.11, 0.1267, 0.1433, 0.16, 0.18, 0.2, 0.22,
    0.2467, 0.2733, 0.3};
static const double DETECT_DELETE_RATES[] = {
    0, 0.0017, 0.0033, 0.005, 0.0067, 0.0083, 0.01, 0.0133, 0.0167, 0.02,
    0.0233, 0.0267, 0.03, 0.0367, 0.0433, 0.05, 0.0567, 0.0633, 0.07, 0.08,
    0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.1733, 0.1867, 0.2,
    0.2167, 0.2333, 0.25};
static const double DETECT_ERASE_RATES[] = {
    0, 0.0033, 0.0067, 0.01, 0.0133, 0.0167, 0.02, 0.0233, 0.0267, 0.03,
    0.0367, 0.0433, 0.05, 0.06, 0.07, 0.08, 0.0933, 0.1067, 0.12, 0.14,
    0.16, 0.18, 0.2033, 0.2267, 0.25, 0.2833, 0.3167, 0.35, 0.3833, 0.4167,
    0.45, 0.5167, 0.5833, 0.65, 0.7167, 0.7833, 0.85, 0.8733, 0.8967, 0.92};

/* Look up the rate grid for a (metric, impairment) pair, with its length via
 * *count. Indexed [metric][axis]; GRID() pairs each array with its own length. */
typedef struct {
  const double *rates;
  int count;
} rate_grid;
#define GRID(arr) {(arr), (int)(sizeof(arr) / sizeof((arr)[0]))}
static const rate_grid GRIDS[N_METRICS][N_AXES] = {
    [METRIC_EDIT] = {[AXIS_FLIP] = GRID(EDIT_FLIP_RATES),
                     [AXIS_INSERT] = GRID(EDIT_INSERT_RATES),
                     [AXIS_DELETE] = GRID(EDIT_DELETE_RATES),
                     [AXIS_ERASE] = GRID(EDIT_ERASE_RATES)},
    [METRIC_LOCK] = {[AXIS_FLIP] = GRID(LOCK_FLIP_RATES),
                     [AXIS_INSERT] = GRID(LOCK_INSERT_RATES),
                     [AXIS_DELETE] = GRID(LOCK_DELETE_RATES),
                     [AXIS_ERASE] = GRID(LOCK_ERASE_RATES)},
    [METRIC_DETECT] = {[AXIS_FLIP] = GRID(DETECT_FLIP_RATES),
                       [AXIS_INSERT] = GRID(DETECT_INSERT_RATES),
                       [AXIS_DELETE] = GRID(DETECT_DELETE_RATES),
                       [AXIS_ERASE] = GRID(DETECT_ERASE_RATES)},
};
#undef GRID

static const double *metric_axis_rates(metric which_metric, axis channel_axis,
                                       int *count) {
  const rate_grid *g = &GRIDS[which_metric][channel_axis];
  *count = g->count;
  return g->rates;
}

/* The decoder model and measurement windows for a code - fixed by the code
 * alone, independent of the axis, rate, and trial, so both the trial worker and
 * the row formatter derive them the same way from make_model(). */
typedef struct {
  dv_stream_params params;
  int code_n;
  int constraint_len;
  int decision_depth;
  int max_drift;
  int warmup;
  int detect_window;
  int detect_step;
} point_model;

/* Per-trial partial sums; summed across a point's trials to form its CSV row. */
typedef struct {
  long long edits, ref_bits, lock_bits, detect_samples;
  double lock_sum, detect_sum;
} trial_result;

static point_model make_model(const dv_code *code) {
  point_model m;
  m.code_n = dv_code_n(code);
  m.constraint_len = dv_code_k(code);

  /* One fixed, channel-agnostic decoder model: every impairment pegged at a flat
   * 1%, regardless of which axis is being swept or how high its rate runs. The
   * decoder is never told what the channel is doing - that is the point. At low
   * rates it sees roughly what it expects; as a rate climbs it meets something
   * increasingly unexpected, which is exactly the stress the sweep is meant to
   * measure. Matching the model to the channel (the old behaviour) let the
   * decoder cheat and made every code look better than it really is. Drift
   * tracking is always on so the pegged insertion/deletion assumption is live on
   * every axis, not just the indel ones. */
  const double pegged = 0.01;
  m.max_drift = 8;
  m.decision_depth = 8 * m.constraint_len;
  m.params = (dv_stream_params){
      .decision_depth = m.decision_depth,
      .max_drift = m.max_drift,
      .p_sub = pegged,
      .p_ins = pegged,
      .p_del = pegged,
      .p_erase = pegged,
  };
  m.warmup = m.decision_depth;

  /* Sliding-window detection. dv_detect collapses a whole buffer into one
   * confidence value, so feeding it the entire stream would yield a single
   * sample per trial. Instead we slide a short window across the received bits
   * and average its detections, the per-window analog of mean_lock's per-bit
   * average - it tracks how recognizable the coded structure stays along the
   * stream. The window clears the code's dv_detect_min_len (30-95 bits here);
   * 512 sits well above it and still gives many windows. Step by half a window
   * so neighbors overlap. */
  m.detect_window = 512;
  const long detect_floor = dv_detect_min_len(m.code_n, m.constraint_len);
  if (detect_floor > 0 && m.detect_window < detect_floor) {
    m.detect_window = (int)detect_floor;
  }
  m.detect_step = m.detect_window / 2;
  return m;
}

/* Run one Monte-Carlo trial for a (code, axis, metric, rate) point with its own
 * PRNG stream seeded from `seed`, returning that trial's partial sums. The
 * metric selects which work the trial does: detection needs only the received
 * buffer (no decode), while edit distance and lock probability both come from a
 * single decode and each skips the other's post-processing. Each trial is fully
 * independent, so trials parallelize across cores (see main). */
static trial_result run_one_trial(const dv_code *code, axis channel_axis,
                                  metric which_metric, double rate,
                                  int info_bits, uint64_t seed) {
  uint64_t rng_state = seed;
  uint64_t *rng = &rng_state;
  const point_model m = make_model(code);

  /* Channel rates: only the active axis is nonzero. */
  const double channel_sub = channel_axis == AXIS_FLIP ? rate : 0.0;
  const double channel_ins = channel_axis == AXIS_INSERT ? rate : 0.0;
  const double channel_del = channel_axis == AXIS_DELETE ? rate : 0.0;
  const double channel_erase = channel_axis == AXIS_ERASE ? rate : 0.0;

  uint8_t *message = xmalloc((size_t)info_bits);
  uint8_t *coded = xmalloc((size_t)((info_bits + m.constraint_len) * m.code_n));

  trial_result r = {0, 0, 0, 0, 0.0, 0.0};
  for (int i = 0; i < info_bits; ++i) {
    message[i] = (uint8_t)(rng_next(rng) & 1u);
  }
  int enc_state = 0, coded_len = 0;
  coded_len += dv_code_encode(code, message, info_bits, &enc_state, coded);
  coded_len += dv_code_encode_flush(code, &enc_state, coded + coded_len);

  uint8_t *received = NULL;
  int received_len = apply_channel(coded, coded_len, channel_sub, channel_ins,
                                   channel_del, channel_erase, rng, &received);

  if (which_metric == METRIC_DETECT) {
    /* Blind detection, averaged over a sliding window: at each step, does this
     * stretch still look like a rate-1/n, k code? dv_detect reads the buffer
     * without modifying it and may report DV_UNDETERMINED (negative) on a window
     * that lands too short; skip those. No decode is needed. */
    for (int p = 0; p + m.detect_window <= received_len; p += m.detect_step) {
      double detect = dv_detect(m.code_n, m.constraint_len, received + p,
                                (size_t)m.detect_window);
      if (detect >= 0.0) {
        r.detect_sum += detect;
        ++r.detect_samples;
      }
    }
    free(received);
    free(message);
    free(coded);
    return r;
  }

  /* Edit distance and lock probability both come from one decode. Lock needs the
   * per-bit lock buffer; edit needs the banded-DP scratch - allocate only what
   * this metric uses. */
  const int decoded_cap = info_bits + 256;
  uint8_t *decoded = xmalloc((size_t)decoded_cap);
  double *lock = which_metric == METRIC_LOCK
                     ? xmalloc((size_t)decoded_cap * sizeof(double))
                     : NULL;

  dv_stream_decoder *decoder = dv_stream_decoder_create(code, &m.params);
  if (!decoder) {
    fprintf(stderr, "dv_metrics: decoder create failed\n");
    exit(1);
  }
  int n_stream = 0;
  int n_decoded = decode_all(decoder, received, received_len, decoded, lock,
                             decoded_cap, &n_stream);
  dv_stream_decoder_destroy(decoder);
  free(received);
  if (n_decoded < 0) {
    fprintf(stderr, "dv_metrics: decode error %d\n", n_decoded);
    exit(1);
  }

  /* Compare against the message after dropping the warm-up transient (from both)
   * and trimming the constraint_len-1 flush bits off the tail. */
  int decoded_end = n_decoded - (m.constraint_len - 1);
  if (decoded_end < m.warmup) {
    decoded_end = m.warmup;
  }
  int decoded_len = decoded_end - m.warmup;
  int ref_len = info_bits - m.warmup;
  if (ref_len > 0 && which_metric == METRIC_EDIT) {
    int *dp_prev = xmalloc((size_t)(decoded_cap + 1) * sizeof(int));
    int *dp_cur = xmalloc((size_t)(decoded_cap + 1) * sizeof(int));
    r.edits = edit_distance(decoded + m.warmup, decoded_len, message + m.warmup,
                            ref_len, dp_prev, dp_cur);
    r.ref_bits = ref_len;
    free(dp_prev);
    free(dp_cur);
  } else if (ref_len > 0) { /* METRIC_LOCK */
    /* Average the decoder's lock probability over the same window. Only
     * streaming bits carry a lock value, so stop at n_stream (the flush tail,
     * already trimmed from the edit window, has none). */
    int lock_end = decoded_end < n_stream ? decoded_end : n_stream;
    for (int i = m.warmup; i < lock_end; ++i) {
      r.lock_sum += lock[i];
      ++r.lock_bits;
    }
  }

  free(message);
  free(coded);
  free(decoded);
  free(lock);
  return r;
}

/* Format the CSV row for a point from its summed trial results. A point measures
 * one metric, so only that metric's value columns are filled; the others are
 * left blank for the plotter to skip. The trailing columns are, in order:
 * ref_bits, edit_distance, edit_rate, lock_bits, mean_lock, detect_samples,
 * mean_detect. */
static void format_row(char *out, size_t cap, const char *code_name,
                       metric which_metric, axis channel_axis, double rate,
                       const point_model *m, int trials, trial_result agg) {
  char head[192];
  snprintf(head, sizeof(head), "%s,%s,%s,%.6g,%.6g,%.6g,%.6g,%.6g,%d,%d,%d",
           code_name, METRIC_NAME[which_metric], AXIS_NAME[channel_axis], rate,
           m->params.p_sub, m->params.p_ins, m->params.p_del, m->params.p_erase,
           m->decision_depth, m->max_drift, trials);
  if (which_metric == METRIC_EDIT) {
    double edit_rate =
        agg.ref_bits > 0 ? (double)agg.edits / (double)agg.ref_bits : 0.0;
    snprintf(out, cap, "%s,%lld,%lld,%.8g,,,,\n", head, agg.ref_bits, agg.edits,
             edit_rate);
  } else if (which_metric == METRIC_LOCK) {
    double mean_lock =
        agg.lock_bits > 0 ? agg.lock_sum / (double)agg.lock_bits : 0.0;
    snprintf(out, cap, "%s,,,,%lld,%.8g,,\n", head, agg.lock_bits, mean_lock);
  } else {
    double mean_detect = agg.detect_samples > 0
                             ? agg.detect_sum / (double)agg.detect_samples
                             : 0.0;
    snprintf(out, cap, "%s,,,,,,%lld,%.8g\n", head, agg.detect_samples,
             mean_detect);
  }
}

int main(int argc, char **argv) {
  /* Many short trials beat one long stream here: the streams self-synchronize,
   * so a single long run would mostly re-measure the same steady state, while
   * independent trials each pay a fresh blind acquisition and sample the
   * heavy-tailed lose-lock events near each code's knee. So we default to a high
   * trial count over a modest message length. */
  int trials = argc > 1 ? atoi(argv[1]) : 50;
  int info_bits = argc > 2 ? atoi(argv[2]) : 1000;
  uint64_t seed = argc > 3 ? strtoull(argv[3], NULL, 0) : 0xC0FFEEULL;
  if (trials < 1 || info_bits < 1) {
    fprintf(stderr, "usage: %s [trials>=1] [info_bits>=1] [seed]\n", argv[0]);
    return 2;
  }

  /* The trellis tables in a dv_code are read-only once built, so all threads
   * share the four codes; each decode allocates its own decoder state. */
  dv_code *codes[N_CODES];
  for (int code_idx = 0; code_idx < N_CODES; ++code_idx) {
    codes[code_idx] = dv_code_create_standard(CODES[code_idx].which);
    if (!codes[code_idx]) {
      fprintf(stderr, "dv_metrics: code create failed\n");
      return 1;
    }
  }

  /* Each (code, axis, rate) point is an independent work item. Axes use
   * different rate grids, so we enumerate the combinations explicitly into a
   * list rather than decomposing a flat index. */
  const int n_axes = N_AXES;
  int n_points = 0;
  for (int metric_idx = 0; metric_idx < N_METRICS; ++metric_idx) {
    for (int axis_idx = 0; axis_idx < n_axes; ++axis_idx) {
      int count;
      metric_axis_rates((metric)metric_idx, (axis)axis_idx, &count);
      n_points += N_CODES * count;
    }
  }

  typedef struct {
    int code_idx;
    axis channel_axis;
    metric which_metric;
    double rate;
  } work_item;
  work_item *items = xmalloc((size_t)n_points * sizeof(*items));
  int filled = 0;
  for (int code_idx = 0; code_idx < N_CODES; ++code_idx) {
    for (int metric_idx = 0; metric_idx < N_METRICS; ++metric_idx) {
      for (int axis_idx = 0; axis_idx < n_axes; ++axis_idx) {
        int count;
        const double *rates =
            metric_axis_rates((metric)metric_idx, (axis)axis_idx, &count);
        for (int rate_idx = 0; rate_idx < count; ++rate_idx) {
          items[filled].code_idx = code_idx;
          items[filled].channel_axis = (axis)axis_idx;
          items[filled].which_metric = (metric)metric_idx;
          items[filled].rate = rates[rate_idx];
          ++filled;
        }
      }
    }
  }

#ifdef _OPENMP
  fprintf(stderr, "running %d points x %d trials on %d threads ...\n", n_points,
          trials, omp_get_max_threads());
#else
  fprintf(stderr, "running %d points x %d trials (single-threaded) ...\n",
          n_points, trials);
#endif

  printf(
      "code,metric,axis,rate,dec_p_sub,dec_p_ins,dec_p_del,dec_p_erase,"
      "decision_depth,max_drift,trials,ref_bits,edit_distance,edit_rate,"
      "lock_bits,mean_lock,detect_samples,mean_detect\n");
  fflush(stdout);

  /* Fan every (point, trial) out as an independent task (collapse(2)), so a
   * single slow point's trials spread across all cores instead of pinning one -
   * the slow drift points would otherwise bound wall-clock. Each trial owns a
   * seed derived from its flat (point, trial) index, so values are reproducible
   * regardless of thread count. Trials write into their own result slots; the
   * task that finishes a point's last trial (tracked by an atomic counter) sums
   * the slots in trial order - keeping mean_lock/mean_detect bit-exact, free of
   * reduction-order wobble - and streams that point's row out (one writer at a
   * time; flush so the file grows as the sweep runs). */
  trial_result *results =
      xmalloc((size_t)n_points * (size_t)trials * sizeof(*results));
  int *remaining = xmalloc((size_t)n_points * sizeof(int));
  for (int point = 0; point < n_points; ++point) {
    remaining[point] = trials;
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) collapse(2)
#endif
  for (int point = 0; point < n_points; ++point) {
    for (int trial = 0; trial < trials; ++trial) {
      const work_item item = items[point];
      results[(size_t)point * trials + trial] =
          run_one_trial(codes[item.code_idx], item.channel_axis,
                        item.which_metric, item.rate, info_bits,
                        derive_seed(seed, point * trials + trial));

      int left;
#ifdef _OPENMP
#pragma omp atomic capture
#endif
      {
        remaining[point] -= 1;
        left = remaining[point];
      }
      if (left == 0) {
        const point_model m = make_model(codes[item.code_idx]);
        trial_result agg = {0, 0, 0, 0, 0.0, 0.0};
        for (int t = 0; t < trials; ++t) {
          const trial_result *s = &results[(size_t)point * trials + t];
          agg.edits += s->edits;
          agg.ref_bits += s->ref_bits;
          agg.lock_bits += s->lock_bits;
          agg.detect_samples += s->detect_samples;
          agg.lock_sum += s->lock_sum;
          agg.detect_sum += s->detect_sum;
        }
        char row[256];
        format_row(row, sizeof(row), CODES[item.code_idx].name,
                   item.which_metric, item.channel_axis, item.rate, &m, trials,
                   agg);
#ifdef _OPENMP
#pragma omp critical
#endif
        {
          fputs(row, stdout);
          fflush(stdout);
        }
      }
    }
  }

  free(results);
  free(remaining);
  free(items);
  for (int code_idx = 0; code_idx < N_CODES; ++code_idx) {
    dv_code_destroy(codes[code_idx]);
  }
  return 0;
}
