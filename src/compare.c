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

#include <drift_viterbi/compare.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* clang-format off */
/*
 * dv_compare: probability that two bit-stream samples were produced by
 * convolutional codes with the *same* generator polynomials, for a given rate
 * 1/n and constraint length k.
 *
 * Method (GF(2) dual-space comparison). A rate-1/n convolutional code is a
 * linear subspace of bit-streams; every valid encoding of *any* input satisfies
 * the same parity-check (dual) relations, which are fixed by the generator
 * polynomials. So "same generators" <=> "same dual space", independent of the
 * transmitted input. For each sample we recover its dual space blindly over
 * GF(2), then measure how well each sample satisfies the other's parity checks.
 *
 * Bit representation follows the drift_viterbi convention: one bit per byte
 * (value & 1); only the low bit of each input byte is used.
 *
 * Drift tolerance: this corrects both a constant framing/phase offset *and*
 * cumulative mid-stream insertion/deletion drift, the kind that misaligns an
 * ordinary frame the further into the stream you read. Two mechanisms cooperate:
 *
 *  - Dual recovery scans short *segments* from the start and takes the earliest
 *    one sitting in a clean run between indels, so the fixed slide-by-n histogram
 *    sees an unmisframed frame. The dual space is global, so checks found there
 *    hold everywhere.
 *  - Cross-satisfaction (dv_cross_satisfaction) then carries those checks across
 *    the whole stream along an offset *path* that may slip by +/-1 bit per window
 *    - a self-synchronising analog of drift_viterbi's drift window / re-anchoring,
 *    bounded to +/-DV_MAX_CUMULATIVE_DRIFT bits of net drift.
 *
 * We do not (and cannot) invoke drift_viterbi's decoder here: that needs a known
 * trellis, whereas this comparison must stay blind to the code.
 */
/* clang-format on */

/* -- tunable constants ----------------------------------------------------- */

/* A candidate parity-check vector joins a sample's dual space if at least this
 * fraction of the sample's windows satisfy it. Clean data sits near 1.0; the
 * margin above 0.5 tolerates substitution noise. */
static const double DV_DUAL_SATISFACTION_THRESHOLD = 0.85;

/* Half-range of the *initial* framing offset searched when matching one
 * sample's checks against the other, absorbing framing phase and a constant
 * skew between captures. The cross-satisfaction path then wanders from here
 * (see below). */
#define DV_MAX_DRIFT 16

/* Bound on *cumulative* drift: the cross-satisfaction offset path may wander up
 * to this many bits from the basis frame over the whole stream (stepping +/-1
 * per window). The cumulative analog of DV_MAX_DRIFT, and the counterpart of
 * the decoder's max_drift. */
#define DV_MAX_CUMULATIVE_DRIFT 128

/* Width of the offset axis in the cross-satisfaction DP (offset in
 * [-DV_MAX_CUMULATIVE_DRIFT, +DV_MAX_CUMULATIVE_DRIFT]). */
#define DV_OFFSET_WIDTH (2 * DV_MAX_CUMULATIVE_DRIFT + 1)

/* Nominal channel priors, used only to weight the offset-path DP - only their
 * relative sizes matter, mirroring the decoder's -log(p) branch metrics. A
 * violated parity check is scored as a substitution; a +/-1 offset step (an
 * inserted/dropped bit in the stream relative to the basis frame) as an indel.
 */
static const double DV_NOMINAL_P_SUB = 0.05;
static const double DV_NOMINAL_P_INDEL = 0.02;

/* Largest relation window (bits). Dual recovery uses a Walsh-Hadamard transform
 * over 2^window_bits bins, so window_bits is capped to bound time and memory.
 */
#define DV_MAX_WINDOW_BITS 22

/* Returned when the result cannot be determined (bad args, too little data, or
 * a code too large for this version). */
#define DV_UNDETERMINED (-1.0)

/* -- helpers --------------------------------------------------------------- */

static double dv_clamp01(double value) {
  if (value < 0.0) return 0.0;
  if (value > 1.0) return 1.0;
  return value;
}

/* Pack window_bits consecutive stream bits (LSB = first bit) into an integer.
 */
static uint32_t dv_window(const uint8_t *stream, size_t start,
                          int window_bits) {
  uint32_t packed = 0;
  for (int i = 0; i < window_bits; ++i) {
    packed |= (uint32_t)(stream[start + (size_t)i] & 1u) << i;
  }
  return packed;
}

/* -- dual-space recovery --------------------------------------------------- */

/*
 * Build the dual-space spectrum of a stream: histogram window_bits-wide windows
 * (slid by n bits) into 2^window_bits bins, then Walsh-Hadamard transform in
 * place. Afterwards spectrum[vector] = (#windows with parity(window & vector)
 * == 0) - (#with parity == 1), so the satisfied fraction of parity vector
 * `vector` is (window_count + spectrum[vector]) / (2 * window_count). Returns a
 * malloc'd array of size 2^window_bits (caller frees) and the window count via
 * *window_count_out.
 *
 * Only the first `max_windows` windows are histogrammed (<= 0 means all of
 * them): restricting recovery to a low-drift prefix keeps cumulative mid-stream
 * drift from misframing the fixed slide-by-n and smearing the spectrum.
 */
static int *dv_dual_spectrum(const uint8_t *stream, size_t len, int n,
                             int window_bits, long max_windows,
                             long *window_count_out) {
  size_t bin_count = (size_t)1 << window_bits;
  long window_count = (long)((len - (size_t)window_bits) / (size_t)n) + 1;
  if (max_windows > 0 && window_count > max_windows) {
    window_count = max_windows;
  }
  int *spectrum = calloc(bin_count, sizeof(*spectrum));
  if (!spectrum) {
    return NULL;
  }

  for (long window_index = 0; window_index < window_count; ++window_index) {
    uint32_t packed =
        dv_window(stream, (size_t)window_index * (size_t)n, window_bits);
    spectrum[packed]++;
  }

  /* In-place Walsh-Hadamard transform: spectrum[vector] = sum over index of
   * spectrum0[index] * (-1)^popcount(index & vector). */
  for (size_t stride = 1; stride < bin_count; stride <<= 1) {
    for (size_t i = 0; i < bin_count; i += stride << 1) {
      for (size_t j = i; j < i + stride; ++j) {
        int lower = spectrum[j];
        int upper = spectrum[j + stride];
        spectrum[j] = lower + upper;
        spectrum[j + stride] = lower - upper;
      }
    }
  }

  *window_count_out = window_count;
  return spectrum;
}

/*
 * Reduce all high-satisfaction parity vectors to a GF(2) basis (one pivot per
 * bit position). Fills basis[] (capacity DV_MAX_WINDOW_BITS) and returns its
 * dimension = the recovered dual space's dimension.
 */
static int dv_dual_basis(const int *spectrum, long window_count,
                         int window_bits, uint32_t *basis) {
  uint32_t row_for_bit[DV_MAX_WINDOW_BITS];
  memset(row_for_bit, 0, sizeof(row_for_bit));

  size_t bin_count = (size_t)1 << window_bits;
  for (size_t vector = 1; vector < bin_count; ++vector) {
    double satisfaction = (double)(window_count + spectrum[vector]) /
                          (2.0 * (double)window_count);
    if (satisfaction < DV_DUAL_SATISFACTION_THRESHOLD) {
      continue;
    }
    uint32_t row = (uint32_t)vector;
    for (int bit = window_bits - 1; bit >= 0; --bit) {
      if (!((row >> bit) & 1u)) {
        continue;
      }
      if (row_for_bit[bit]) {
        row ^= row_for_bit[bit];
      } else {
        row_for_bit[bit] = row;
        break;
      }
    }
  }

  int dimension = 0;
  for (int bit = 0; bit < window_bits; ++bit) {
    if (row_for_bit[bit]) {
      basis[dimension++] = row_for_bit[bit];
    }
  }
  return dimension;
}

/* -- cross-satisfaction ---------------------------------------------------- */

/* Satisfied-check count of one window_bits-wide window of the stream at byte
 * position `position` against every parity vector in basis[]. */
static int dv_window_good(const uint8_t *stream, long position, int window_bits,
                          const uint32_t *basis, int dimension) {
  uint32_t packed = dv_window(stream, (size_t)position, window_bits);
  int good = 0;
  for (int i = 0; i < dimension; ++i) {
    if (!__builtin_parity(packed & basis[i])) {
      ++good;
    }
  }
  return good;
}

/* clang-format off */
/*
 * Drift-tolerant cross-satisfaction: the best achievable fraction of the
 * stream's parity checks satisfied when the framing offset is allowed to follow
 * a smooth path that slips by +/-1 bit per window (a bit inserted into / dropped
 * from the stream relative to the basis frame). ~1.0 when the stream shares the
 * code, ~0.5 when it does not.
 *
 * This is a traceback-free Viterbi over the offset state `offset` in
 * [-DV_MAX_CUMULATIVE_DRIFT, +DV_MAX_CUMULATIVE_DRIFT]: window window_index reads
 * window_bits bits of the stream at position window_index*n + offset, and we
 * pick the offset trajectory of least total cost, where
 *   cost = sum [ violated*cost_miss + satisfied*cost_match ] (per-window misfit)
 *        + sum [ offset stepped ? cost_indel : 0 ]           (per-step penalty)
 * in negative-log-likelihood units (mirroring the decoder's branch metrics).
 * Window 0 may start anywhere within +/-DV_MAX_DRIFT for free, absorbing framing
 * phase and a constant skew.
 *
 * The indel penalty is the load-bearing property: without it a free-wandering
 * offset would cherry-pick a locally-satisfying frame at every window and
 * manufacture satisfaction on an unrelated stream. With it, the path must be
 * smooth, so only genuinely shared structure scores high.
 *
 * Every full-length path processes the same window_count windows of `dimension`
 * checks each, so the denominator window_count*dimension is path-independent; we
 * carry only the satisfied count (good) along the winning path and return
 * good / (window_count * dimension).
 */
/* clang-format on */
static double dv_cross_satisfaction(const uint8_t *stream, size_t len, int n,
                                    int window_bits, const uint32_t *basis,
                                    int dimension) {
  const int max_offset = DV_MAX_CUMULATIVE_DRIFT;
  const int offset_width = DV_OFFSET_WIDTH;
  const long window_count = (long)((len - (size_t)window_bits) / (size_t)n) +
                            1; /* offset=0 windows */
  if (window_count < 1) {
    return 0.0;
  }

  const double cost_match = -log(1.0 - DV_NOMINAL_P_SUB);
  const double cost_miss = -log(DV_NOMINAL_P_SUB);
  const double cost_indel = -log(DV_NOMINAL_P_INDEL);

  /* Two rolling rows of {cost, good}; offset_index maps to offset =
   * offset_index - max_offset. offset_width is a compile-time constant, so
   * these live on the stack. */
  double cost[DV_OFFSET_WIDTH], next_cost[DV_OFFSET_WIDTH];
  long good[DV_OFFSET_WIDTH], next_good[DV_OFFSET_WIDTH];

  /* Window 0: any starting offset within +/-DV_MAX_DRIFT is free. */
  for (int offset_index = 0; offset_index < offset_width; ++offset_index) {
    const int offset = offset_index - max_offset;
    cost[offset_index] = INFINITY;
    good[offset_index] = 0;
    if (abs(offset) > DV_MAX_DRIFT || offset < 0 ||
        offset + window_bits > (long)len) {
      continue;
    }
    const int good_count =
        dv_window_good(stream, offset, window_bits, basis, dimension);
    good[offset_index] = good_count;
    cost[offset_index] = (double)(dimension - good_count) * cost_miss +
                         (double)good_count * cost_match;
  }

  for (long window_index = 1; window_index < window_count; ++window_index) {
    const long window_base = window_index * (long)n;
    for (int offset_index = 0; offset_index < offset_width; ++offset_index) {
      const long position = window_base + (offset_index - max_offset);
      next_cost[offset_index] = INFINITY;
      next_good[offset_index] = 0;
      if (position < 0 || position + window_bits > (long)len) {
        continue;
      }
      /* Best predecessor among offset, offset-1, offset+1 (a +/-1 step pays
       * cost_indel). */
      double best = INFINITY;
      long best_good = 0;
      for (int step = -1; step <= 1; ++step) {
        const int predecessor = offset_index + step;
        if (predecessor < 0 || predecessor >= offset_width ||
            cost[predecessor] == INFINITY) {
          continue;
        }
        const double transition =
            cost[predecessor] + (step == 0 ? 0.0 : cost_indel);
        if (transition < best) {
          best = transition;
          best_good = good[predecessor];
        }
      }
      if (best == INFINITY) {
        continue;
      }
      const int good_count =
          dv_window_good(stream, position, window_bits, basis, dimension);
      next_cost[offset_index] = best +
                                (double)(dimension - good_count) * cost_miss +
                                (double)good_count * cost_match;
      next_good[offset_index] = best_good + good_count;
    }
    memcpy(cost, next_cost, sizeof(cost));
    memcpy(good, next_good, sizeof(good));
  }

  double best = INFINITY;
  long best_good = 0;
  for (int offset_index = 0; offset_index < offset_width; ++offset_index) {
    if (cost[offset_index] < best) {
      best = cost[offset_index];
      best_good = good[offset_index];
    }
  }
  if (best == INFINITY) {
    return 0.0; /* no feasible full-length path */
  }
  return (double)best_good / ((double)window_count * (double)dimension);
}

/* -- basis recovery -------------------------------------------------------- */

/* A candidate basis is accepted only if it explains its own stream at least
 * this well (drift-tolerant self cross-satisfaction). A correct basis
 * self-scores ~1.0; a spurious one recovered from a misframed or noisy segment
 * scores ~0.5, and an unstructured (random) stream stays below this floor
 * entirely. */
static const double DV_SELF_SATISFACTION_FLOOR = 0.9;

/* Self cross-satisfaction this high means the candidate clearly is the stream's
 * dual space; stop scanning further segments. */
static const double DV_SELF_SATISFACTION_GOOD = 0.97;

/*
 * Recover a stream's dual basis into basis[] (capacity DV_MAX_WINDOW_BITS) and
 * return its dimension. Returns -1 on allocation failure, 0 if no reliable
 * basis emerged (e.g. an unstructured stream).
 *
 * The fixed slide-by-n histogram needs a clean frame, but cumulative drift
 * misframes it: under deletions a single dropped bit shifts every later window,
 * so a span straddling one deletion smears the spectrum. We therefore scan
 * SEGMENTS forward from the start - sliding a window until it lands in a clean
 * run between indels - and recover a candidate basis from each. The dual space
 * is global, so a basis from any clean run is valid everywhere;
 * cross-satisfaction then carries it across the drifted remainder.
 *
 * Each candidate is self-validated by its drift-tolerant cross-satisfaction
 * against the whole stream, which cheaply rejects spurious vectors that a short
 * segment can admit: we keep the best-scoring candidate and stop early once one
 * clearly fits (DV_SELF_SATISFACTION_GOOD). A clean stream is validated at the
 * first segment, preserving the original behaviour.
 */
static int recover_basis(const uint8_t *stream, size_t len, int n,
                         int window_bits, uint32_t *basis) {
  /* Segment window count: large enough to pin the dual space without admitting
   * spurious checks, short enough to fit inside a typical clean run. */
  long segment_windows = 4L * window_bits;
  if (segment_windows < window_bits + 8) {
    segment_windows = window_bits + 8;
  }
  const long total_windows =
      (long)((len - (size_t)window_bits) / (size_t)n) + 1;

  uint32_t candidate[DV_MAX_WINDOW_BITS];
  int best_dimension = 0;
  double best_self_satisfaction = 0.0;
  for (long start = 0; start + (window_bits + 1) <= total_windows;
       start += segment_windows) {
    const size_t offset = (size_t)start * (size_t)n;
    long used = 0;
    int *spectrum = dv_dual_spectrum(stream + offset, len - offset, n,
                                     window_bits, segment_windows, &used);
    if (!spectrum) {
      return -1;
    }
    const int dimension =
        (used >= window_bits + 1)
            ? dv_dual_basis(spectrum, used, window_bits, candidate)
            : 0;
    free(spectrum);
    if (dimension <= 0) {
      continue;
    }
    const double self_satisfaction = dv_cross_satisfaction(
        stream, len, n, window_bits, candidate, dimension);
    if (self_satisfaction > best_self_satisfaction) {
      best_self_satisfaction = self_satisfaction;
      best_dimension = dimension;
      memcpy(basis, candidate, (size_t)dimension * sizeof(*candidate));
      if (self_satisfaction >= DV_SELF_SATISFACTION_GOOD) {
        break;
      }
    }
  }
  return best_self_satisfaction >= DV_SELF_SATISFACTION_FLOOR ? best_dimension
                                                              : 0;
}

/* -- public API ------------------------------------------------------------ */

double dv_compare(int n, int k, uint8_t *lhs, size_t lhs_len, uint8_t *rhs,
                  size_t rhs_len) {
  if (n < 1 || k < 2 || k > 9 || !lhs || !rhs) {
    return DV_UNDETERMINED;
  }

  int window_bits = n * (k + 1);
  if (window_bits < 1 || window_bits > DV_MAX_WINDOW_BITS) {
    return DV_UNDETERMINED;
  }
  if (lhs_len < (size_t)window_bits || rhs_len < (size_t)window_bits) {
    return DV_UNDETERMINED;
  }

  /* Need clearly more windows than the window width to pin down the dual space.
   */
  long windows_lhs = (long)((lhs_len - (size_t)window_bits) / (size_t)n) + 1;
  long windows_rhs = (long)((rhs_len - (size_t)window_bits) / (size_t)n) + 1;
  if (windows_lhs < window_bits + 1 || windows_rhs < window_bits + 1) {
    return DV_UNDETERMINED;
  }

  uint32_t basis_lhs[DV_MAX_WINDOW_BITS];
  uint32_t basis_rhs[DV_MAX_WINDOW_BITS];
  int dimension_lhs = recover_basis(lhs, lhs_len, n, window_bits, basis_lhs);
  int dimension_rhs = recover_basis(rhs, rhs_len, n, window_bits, basis_rhs);
  if (dimension_lhs < 0 || dimension_rhs < 0) {
    return DV_UNDETERMINED; /* allocation failure */
  }

  if (dimension_lhs == 0 && dimension_rhs == 0) {
    return DV_UNDETERMINED; /* neither sample exposed any linear structure */
  }
  if (dimension_lhs == 0 || dimension_rhs == 0) {
    return 0.0; /* one is structured, the other is not -> different */
  }

  /*
   * Test each sample's parity checks against the other (drift-tolerant). The
   * cross-satisfaction is the membership test: ~1 when both samples obey the
   * same dual space, ~0.5 (-> 0 after rescaling) when they do not. We take the
   * weaker direction so a one-sided coincidence cannot inflate the result.
   *
   * We deliberately do NOT penalize a difference in recovered dual dimension:
   * a framing/phase offset between the two samples changes the null-space
   * dimension at the window boundary even for an identical code.
   */
  double cross_lhs_on_rhs =
      dv_clamp01(2.0 * (dv_cross_satisfaction(rhs, rhs_len, n, window_bits,
                                              basis_lhs, dimension_lhs) -
                        0.5));
  double cross_rhs_on_lhs =
      dv_clamp01(2.0 * (dv_cross_satisfaction(lhs, lhs_len, n, window_bits,
                                              basis_rhs, dimension_rhs) -
                        0.5));

  return (cross_lhs_on_rhs < cross_rhs_on_lhs) ? cross_lhs_on_rhs
                                               : cross_rhs_on_lhs;
}
