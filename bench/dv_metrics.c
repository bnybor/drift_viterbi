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
 * Output is CSV on stdout (see header row); feed it to bench/plot_metrics.py.
 *
 * Usage: dv_metrics [trials] [info_bits] [seed]
 */

#include "drift_viterbi/drift_viterbi.h"

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

static double max_double(double a, double b) { return a > b ? a : b; }

/* Push a received buffer through the streaming decoder in small chunks, then
 * drain. Returns the number of decoded bits collected (<= decoded_cap). */
static int decode_all(dv_stream_decoder *decoder, const uint8_t *received,
                      int received_len, uint8_t *decoded, int decoded_cap) {
  int n_decoded = 0, read_pos = 0;
  while (read_pos < received_len && n_decoded < decoded_cap) {
    int chunk = received_len - read_pos < 64 ? received_len - read_pos : 64;
    int written = dv_stream_decode(decoder, received + read_pos, chunk,
                                   decoded + n_decoded, decoded_cap - n_decoded);
    if (written < 0) {
      return written;
    }
    n_decoded += written;
    read_pos += chunk;
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

/* Channel rates to sweep along whichever axis is active. Denser near the low
 * end, where the codes' error-correction knees sit. */
static const double RATES[] = {
    0.0,   0.001,  0.002, 0.003,  0.004, 0.005, 0.006, 0.0075,
    0.01,  0.0125, 0.015, 0.0175, 0.02,  0.025, 0.03,  0.035,
    0.04,  0.05,   0.06,  0.07,   0.08,  0.09,  0.10,  0.11,
    0.12,  0.13,   0.14,  0.15,   0.16,  0.17,  0.18,  0.19,
    0.20};
#define N_RATES ((int)(sizeof(RATES) / sizeof(RATES[0])))

/* Erasures are far more correctable than the other impairments - a rate-1/n
 * code only fails as the erasure rate nears its (1 - rate) limit - so the erase
 * axis sweeps a much wider range to reach the stronger codes' knees. */
static const double ERASE_RATES[] = {
    0.0,  0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14,
    0.16, 0.18, 0.20, 0.23, 0.26, 0.29, 0.32, 0.35,
    0.38, 0.41, 0.44, 0.47, 0.50, 0.53, 0.56, 0.59,
    0.62, 0.65, 0.68, 0.71, 0.74, 0.77, 0.80};
#define N_ERASE_RATES ((int)(sizeof(ERASE_RATES) / sizeof(ERASE_RATES[0])))

typedef enum { AXIS_FLIP, AXIS_INSERT, AXIS_DELETE, AXIS_ERASE } axis;
static const char *AXIS_NAME[] = {"flip", "insert", "delete", "erase"};

/* The rate grid to sweep for an axis, and its length via *count. */
static const double *axis_rates(axis channel_axis, int *count) {
  if (channel_axis == AXIS_ERASE) {
    *count = N_ERASE_RATES;
    return ERASE_RATES;
  }
  *count = N_RATES;
  return RATES;
}

/* Run `trials` Monte-Carlo trials for one (code, axis, rate) point, summing the
 * edit distance and compared bits, and format its CSV row into `out_row`. Uses
 * its own PRNG stream seeded from `seed`, so points are independent. */
static void run_point(const dv_code *code, const char *code_name,
                      axis channel_axis, double rate, int trials, int info_bits,
                      uint64_t seed, char *out_row, size_t out_row_cap) {
  uint64_t rng_state = seed;
  uint64_t *rng = &rng_state;
  const int code_n = dv_code_n(code);
  const int constraint_len = dv_code_k(code);

  /* Channel rates: only the active axis is nonzero. */
  double channel_sub = channel_axis == AXIS_FLIP ? rate : 0.0;
  double channel_ins = channel_axis == AXIS_INSERT ? rate : 0.0;
  double channel_del = channel_axis == AXIS_DELETE ? rate : 0.0;
  double channel_erase = channel_axis == AXIS_ERASE ? rate : 0.0;

  /* Decoder model. Only insertions and deletions shift timing, so drift is
   * tracked just on those axes (max_drift 0 for flips and erasures). The active
   * axis's probability drives the model; the rest are floored to stay strictly
   * positive, which the decoder requires - rough values are fine, only relative
   * sizes matter. */
  const double min_prob = 1e-3;
  int max_drift = (channel_axis == AXIS_INSERT || channel_axis == AXIS_DELETE)
                      ? 8
                      : 0;
  int decision_depth = 8 * constraint_len;
  dv_stream_params params = {
      .decision_depth = decision_depth,
      .max_drift = max_drift,
      .p_sub = (channel_axis == AXIS_FLIP) ? max_double(rate, min_prob) : 0.005,
      .p_ins = (max_drift > 0) ? max_double(channel_ins, min_prob) : 0.0,
      .p_del = (max_drift > 0) ? max_double(channel_del, min_prob) : 0.0,
      .p_erase = (channel_axis == AXIS_ERASE) ? max_double(rate, min_prob) : 0.0,
  };
  const int warmup = decision_depth;

  const int coded_cap = (info_bits + constraint_len) * code_n;
  const int decoded_cap = info_bits + 256;
  uint8_t *message = xmalloc((size_t)info_bits);
  uint8_t *coded = xmalloc((size_t)coded_cap);
  uint8_t *decoded = xmalloc((size_t)decoded_cap);
  int *dp_prev = xmalloc((size_t)(decoded_cap + 1) * sizeof(int));
  int *dp_cur = xmalloc((size_t)(decoded_cap + 1) * sizeof(int));

  long long edits = 0, ref_bits = 0;
  for (int trial = 0; trial < trials; ++trial) {
    for (int i = 0; i < info_bits; ++i) {
      message[i] = (uint8_t)(rng_next(rng) & 1u);
    }
    int enc_state = 0, coded_len = 0;
    coded_len += dv_code_encode(code, message, info_bits, &enc_state, coded);
    coded_len += dv_code_encode_flush(code, &enc_state, coded + coded_len);

    uint8_t *received = NULL;
    int received_len = apply_channel(coded, coded_len, channel_sub, channel_ins,
                                     channel_del, channel_erase, rng, &received);

    dv_stream_decoder *decoder = dv_stream_decoder_create(code, &params);
    if (!decoder) {
      fprintf(stderr, "dv_metrics: decoder create failed\n");
      exit(1);
    }
    int n_decoded =
        decode_all(decoder, received, received_len, decoded, decoded_cap);
    dv_stream_decoder_destroy(decoder);
    free(received);
    if (n_decoded < 0) {
      fprintf(stderr, "dv_metrics: decode error %d\n", n_decoded);
      exit(1);
    }

    /* Compare the decoded bits to the message after dropping the warm-up
     * transient (from both) and trimming the constraint_len-1 flush bits off
     * the tail. */
    int decoded_end = n_decoded - (constraint_len - 1);
    if (decoded_end < warmup) {
      decoded_end = warmup;
    }
    int decoded_len = decoded_end - warmup;
    int ref_len = info_bits - warmup;
    if (ref_len <= 0) {
      continue;
    }
    edits += edit_distance(decoded + warmup, decoded_len, message + warmup,
                           ref_len, dp_prev, dp_cur);
    ref_bits += ref_len;
  }

  double edit_rate = ref_bits > 0 ? (double)edits / (double)ref_bits : 0.0;
  snprintf(out_row, out_row_cap,
           "%s,%s,%.6g,%.6g,%.6g,%.6g,%.6g,%d,%d,%d,%lld,%lld,%.8g\n", code_name,
           AXIS_NAME[channel_axis], rate, params.p_sub, params.p_ins,
           params.p_del, params.p_erase, decision_depth, max_drift, trials,
           ref_bits, edits, edit_rate);

  free(message);
  free(coded);
  free(decoded);
  free(dp_prev);
  free(dp_cur);
}

int main(int argc, char **argv) {
  int trials = argc > 1 ? atoi(argv[1]) : 12;
  int info_bits = argc > 2 ? atoi(argv[2]) : 4000;
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
  const int n_axes = AXIS_ERASE + 1;
  int n_points = 0;
  for (int axis_idx = 0; axis_idx < n_axes; ++axis_idx) {
    int count;
    axis_rates((axis)axis_idx, &count);
    n_points += N_CODES * count;
  }

  typedef struct {
    int code_idx;
    axis channel_axis;
    double rate;
  } work_item;
  work_item *items = xmalloc((size_t)n_points * sizeof(*items));
  int filled = 0;
  for (int code_idx = 0; code_idx < N_CODES; ++code_idx) {
    for (int axis_idx = 0; axis_idx < n_axes; ++axis_idx) {
      int count;
      const double *rates = axis_rates((axis)axis_idx, &count);
      for (int rate_idx = 0; rate_idx < count; ++rate_idx) {
        items[filled].code_idx = code_idx;
        items[filled].channel_axis = (axis)axis_idx;
        items[filled].rate = rates[rate_idx];
        ++filled;
      }
    }
  }

  /* Fan the points out across threads and store each row, then print in list
   * order so the CSV is identical regardless of thread count. */
  const size_t row_cap = 256;
  char *rows = xmalloc((size_t)n_points * row_cap);

#ifdef _OPENMP
  fprintf(stderr, "running %d points on %d threads ...\n", n_points,
          omp_get_max_threads());
#else
  fprintf(stderr, "running %d points (single-threaded) ...\n", n_points);
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int point = 0; point < n_points; ++point) {
    const work_item item = items[point];
    run_point(codes[item.code_idx], CODES[item.code_idx].name,
              item.channel_axis, item.rate, trials, info_bits,
              derive_seed(seed, point), rows + (size_t)point * row_cap, row_cap);
  }

  printf(
      "code,axis,rate,dec_p_sub,dec_p_ins,dec_p_del,dec_p_erase,decision_depth,"
      "max_drift,trials,ref_bits,edit_distance,edit_rate\n");
  for (int point = 0; point < n_points; ++point) {
    fputs(rows + (size_t)point * row_cap, stdout);
  }

  free(rows);
  free(items);
  for (int code_idx = 0; code_idx < N_CODES; ++code_idx) {
    dv_code_destroy(codes[code_idx]);
  }
  return 0;
}
