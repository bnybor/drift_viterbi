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

#include <drift_viterbi/stdlib.h>

#include <math.h>

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
 * relative sizes matter, mirroring the decoder's -dv_log(p) branch metrics. A
 * violated parity check is scored as a substitution; a +/-1 offset step (an
 * inserted/dropped bit in the stream relative to the basis frame) as an indel.
 */
static const double DV_NOMINAL_P_SUB = 0.05;
static const double DV_NOMINAL_P_INDEL = 0.02;

/* Dual recovery is hybrid by window width. For window_bits <=
 * DV_MAX_WINDOW_BITS it uses a Walsh-Hadamard transform over 2^window_bits bins
 * (noise-robust but exponential, so the width is capped). Wider windows use an
 * exact GF(2) null-space recovery instead (recover_segment_nullspace), which is
 * linear in the window count and O(window_bits) in memory. */
#define DV_MAX_WINDOW_BITS 22

/* Hard ceiling on the relation window: parity vectors are packed into a
 * uint32_t, so window_bits must fit. Codes whose window n*(k+1) exceeds this
 * are undetermined (widening the bitvector to uint64_t would raise it to 64).
 */
#define DV_HARD_WINDOW_CAP 32

/* Returned when the result cannot be determined (bad args, too little data, or
 * a code whose relation window exceeds DV_HARD_WINDOW_CAP). */
#define DV_UNDETERMINED (-1.0)

/* Cap on the windows the cross-satisfaction DP processes, bounding time on very
 * long streams; the dual space is global, so a prefix this size suffices. */
#define DV_MAX_CROSS_WINDOWS 200000L

/* Fewest windows a stream needs before recovery is attempted: roughly enough
 * independent step-windows (one per n bits) to pin the dual space, plus margin.
 * Below this dv_compare is undetermined; above it, a stream that is still too
 * thin is caught by self-validation (see recover_basis). */
static long dv_min_recovery_windows(int n, int window_bits) {
  long floor = (long)(window_bits / n) + 8;
  return floor < 8 ? 8 : floor;
}

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
  int *spectrum = dv_calloc(bin_count, sizeof(*spectrum));
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
 * bit position). Fills basis[] (capacity DV_HARD_WINDOW_CAP) and returns its
 * dimension = the recovered dual space's dimension.
 */
static int dv_dual_basis(const int *spectrum, long window_count,
                         int window_bits, uint32_t *basis) {
  uint32_t row_for_bit[DV_MAX_WINDOW_BITS];
  dv_memset(row_for_bit, 0, sizeof(row_for_bit));

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

/*
 * Recover a dual basis from a segment WITHOUT enumerating 2^window_bits, for
 * windows too wide for the WHT. The observed windows are codewords, so every
 * parity vector is orthogonal (over GF(2)) to all of them: the dual space is
 * the orthogonal complement of their span. We row-reduce the windows into a
 * basis of the code (row) space, then read its null space straight off the
 * reduced form. Fills basis[] (capacity DV_HARD_WINDOW_CAP) with up to
 * window_bits - rank vectors and returns that count. Memory is O(window_bits);
 * no allocation.
 *
 * The candidates are orthogonal to every window of THIS segment by
 * construction, so they need no local satisfaction filter; a segment corrupted
 * by a flip (or too short to span the code space) yields spurious vectors that
 * the caller's whole-stream self cross-satisfaction then rejects, exactly as
 * for the WHT path.
 */
static int recover_segment_nullspace(const uint8_t *stream, size_t len, int n,
                                     int window_bits, long max_windows,
                                     uint32_t *basis) {
  long window_count = (long)((len - (size_t)window_bits) / (size_t)n) + 1;
  if (max_windows > 0 && window_count > max_windows) {
    window_count = max_windows;
  }

  /* Row-echelon basis of the code space: pivot[bit] (if set) has its leading 1
   * at `bit`. Standard GF(2) elimination, the same idiom as dv_dual_basis. */
  uint32_t pivot[DV_HARD_WINDOW_CAP];
  dv_memset(pivot, 0, sizeof(pivot));
  for (long window_index = 0; window_index < window_count; ++window_index) {
    uint32_t row =
        dv_window(stream, (size_t)window_index * (size_t)n, window_bits);
    for (int bit = window_bits - 1; bit >= 0 && row; --bit) {
      if (!((row >> bit) & 1u)) {
        continue;
      }
      if (pivot[bit]) {
        row ^= pivot[bit];
      } else {
        pivot[bit] = row;
        break;
      }
    }
  }

  /* Reduce to RREF: clear each pivot column from every other pivot row, so a
   * pivot bit is set only in its own row. */
  for (int bit = 0; bit < window_bits; ++bit) {
    if (!pivot[bit]) {
      continue;
    }
    for (int other = 0; other < window_bits; ++other) {
      if (other != bit && pivot[other] && ((pivot[other] >> bit) & 1u)) {
        pivot[other] ^= pivot[bit];
      }
    }
  }

  /* Null space: one vector per free column. v_f has bit f set, and for every
   * pivot row (pivot column p) whose reduced row has a 1 in column f, bit p
   * set. Then row . v_f == 0 for every code-space row, so v_f is a parity
   * check. */
  int dimension = 0;
  for (int free_col = 0; free_col < window_bits; ++free_col) {
    if (pivot[free_col]) {
      continue; /* a pivot column, not free */
    }
    uint32_t v = (uint32_t)1u << free_col;
    for (int bit = 0; bit < window_bits; ++bit) {
      if (pivot[bit] && ((pivot[bit] >> free_col) & 1u)) {
        v |= (uint32_t)1u << bit;
      }
    }
    basis[dimension++] = v;
  }
  return dimension;
}

/* Recover a candidate dual basis from one segment, dispatching by window width:
 * the noise-robust Walsh-Hadamard path for narrow windows, the scalable GF(2)
 * null-space path for wider ones. Returns the dimension, or -1 on allocation
 * failure (WHT path only). */
static int recover_segment(const uint8_t *stream, size_t len, int n,
                           int window_bits, long max_windows, uint32_t *basis) {
  if (window_bits <= DV_MAX_WINDOW_BITS) {
    long used = 0;
    int *spectrum =
        dv_dual_spectrum(stream, len, n, window_bits, max_windows, &used);
    if (!spectrum) {
      return -1;
    }
    const int dimension =
        (used >= dv_min_recovery_windows(n, window_bits))
            ? dv_dual_basis(spectrum, used, window_bits, basis)
            : 0;
    dv_free(spectrum);
    return dimension;
  }
  return recover_segment_nullspace(stream, len, n, window_bits, max_windows,
                                   basis);
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
  long window_count = (long)((len - (size_t)window_bits) / (size_t)n) +
                      1; /* offset=0 windows */
  if (window_count < 1) {
    return 0.0;
  }
  /* Bound work on very long streams: the dual space is global, so the satisfied
   * fraction over a long prefix represents the whole stream. */
  if (window_count > DV_MAX_CROSS_WINDOWS) {
    window_count = DV_MAX_CROSS_WINDOWS;
  }

  const double cost_match = -dv_log(1.0 - DV_NOMINAL_P_SUB);
  const double cost_miss = -dv_log(DV_NOMINAL_P_SUB);
  const double cost_indel = -dv_log(DV_NOMINAL_P_INDEL);

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
    if (dv_abs(offset) > DV_MAX_DRIFT || offset < 0 ||
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
    dv_memcpy(cost, next_cost, sizeof(cost));
    dv_memcpy(good, next_good, sizeof(good));
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
 * Recover a stream's dual basis into basis[] (capacity DV_HARD_WINDOW_CAP) and
 * return its dimension. Returns -1 on allocation failure, 0 if no reliable
 * basis emerged (e.g. an unstructured stream). When self_satisfaction_out is
 * non-NULL it receives the best candidate's whole-stream self
 * cross-satisfaction
 * (~1.0 for a clean coded stream, ~0.5 for noise), a graded measure of how well
 * the stream fits a code - used by dv_detect.
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
                         int window_bits, uint32_t *basis,
                         double *self_satisfaction_out) {
  const long min_windows = dv_min_recovery_windows(n, window_bits);
  /* Segment window count: large enough to pin the dual space without admitting
   * spurious checks, short enough to fit inside a typical clean run. */
  long segment_windows = 4L * window_bits;
  if (segment_windows < window_bits + 8) {
    segment_windows = window_bits + 8;
  }
  const long total_windows =
      (long)((len - (size_t)window_bits) / (size_t)n) + 1;

  uint32_t candidate[DV_HARD_WINDOW_CAP];
  int best_dimension = 0;
  double best_self_satisfaction = 0.0;
  for (long start = 0; start + min_windows <= total_windows;
       start += segment_windows) {
    const size_t offset = (size_t)start * (size_t)n;
    const int dimension =
        recover_segment(stream + offset, len - offset, n, window_bits,
                        segment_windows, candidate);
    if (dimension < 0) {
      return -1; /* allocation failure */
    }
    if (dimension == 0) {
      continue;
    }
    const double self_satisfaction = dv_cross_satisfaction(
        stream, len, n, window_bits, candidate, dimension);
    if (self_satisfaction > best_self_satisfaction) {
      best_self_satisfaction = self_satisfaction;
      best_dimension = dimension;
      dv_memcpy(basis, candidate, (size_t)dimension * sizeof(*candidate));
      if (self_satisfaction >= DV_SELF_SATISFACTION_GOOD) {
        break;
      }
    }
  }
  if (self_satisfaction_out) {
    *self_satisfaction_out = best_self_satisfaction;
  }
  return best_self_satisfaction >= DV_SELF_SATISFACTION_FLOOR ? best_dimension
                                                              : 0;
}

/* -- public API ------------------------------------------------------------ */

/* Relation window width for a (rate 1/n, constraint length k) code, or 0 if the
 * code is outside the range dv_compare can handle (n < 1, k < 2, k > 9, or a
 * window wider than DV_HARD_WINDOW_CAP). Shared by dv_compare and the length
 * helpers so they agree on what is in range. */
static int dv_window_bits(int n, int k) {
  if (n < 1 || k < 2 || k > 9) {
    return 0;
  }
  const int window_bits = n * (k + 1);
  return (window_bits >= 1 && window_bits <= DV_HARD_WINDOW_CAP) ? window_bits
                                                                 : 0;
}

long dv_compare_min_len(int n, int k) {
  const int window_bits = dv_window_bits(n, k);
  if (window_bits == 0) {
    return -1;
  }
  /* Smallest length whose window count reaches the recovery floor: window count
   * is (len - window_bits) / n + 1, so this hits exactly
   * dv_min_recovery_windows (one bit shorter falls below it and dv_compare is
   * undetermined). */
  const long min_windows = dv_min_recovery_windows(n, window_bits);
  return (long)window_bits + (min_windows - 1) * (long)n;
}

long dv_compare_max_len(int n, int k) {
  const int window_bits = dv_window_bits(n, k);
  if (window_bits == 0) {
    return -1;
  }
  /* Length at which the cross-satisfaction window cap is reached; beyond it the
   * scoring consults only this prefix, so a longer sample cannot change the
   * result and may be truncated to this length. */
  return (long)window_bits + (DV_MAX_CROSS_WINDOWS - 1) * (long)n;
}

/* dv_detect runs the same single-stream code recovery that dv_compare applies
 * to each of its two inputs, so it has the same per-sample length requirements.
 */
long dv_detect_min_len(int n, int k) { return dv_compare_min_len(n, k); }
long dv_detect_max_len(int n, int k) { return dv_compare_max_len(n, k); }

double dv_compare(int n, int k, uint8_t *lhs, size_t lhs_len, uint8_t *rhs,
                  size_t rhs_len) {
  if (!lhs || !rhs) {
    return DV_UNDETERMINED;
  }

  int window_bits = dv_window_bits(n, k);
  if (window_bits == 0) {
    return DV_UNDETERMINED;
  }
  if (lhs_len < (size_t)window_bits || rhs_len < (size_t)window_bits) {
    return DV_UNDETERMINED;
  }

  /* Need enough windows to pin down the dual space; below this floor there is
   * too little data to recover it (anything still too thin self-validates away
   * inside recover_basis). */
  const long min_windows = dv_min_recovery_windows(n, window_bits);
  long windows_lhs = (long)((lhs_len - (size_t)window_bits) / (size_t)n) + 1;
  long windows_rhs = (long)((rhs_len - (size_t)window_bits) / (size_t)n) + 1;
  if (windows_lhs < min_windows || windows_rhs < min_windows) {
    return DV_UNDETERMINED;
  }

  uint32_t basis_lhs[DV_HARD_WINDOW_CAP];
  uint32_t basis_rhs[DV_HARD_WINDOW_CAP];
  int dimension_lhs =
      recover_basis(lhs, lhs_len, n, window_bits, basis_lhs, NULL);
  int dimension_rhs =
      recover_basis(rhs, rhs_len, n, window_bits, basis_rhs, NULL);
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

/*
 * Detect whether a single buffer carries any rate-1/n, constraint-length-k
 * convolutional code, without identifying which one. Same recovery pipeline as
 * one side of dv_compare: blindly recover the stream's own dual (parity-check)
 * space and measure how well the stream satisfies it. A genuine coded stream
 * obeys its checks (~1), random data does not (~0). Indel tolerance comes from
 * the drift-tolerant offset path in cross-satisfaction and erasure/substitution
 * tolerance from the segment scan, exactly as in dv_compare.
 *
 * Returns the probability in [0, 1], or a negative value when it cannot be
 * determined (null buffer, out-of-range code, or too little data).
 */
double dv_detect(int n, int k, uint8_t *sample, size_t len) {
  if (!sample) {
    return DV_UNDETERMINED;
  }
  const int window_bits = dv_window_bits(n, k);
  if (window_bits == 0) {
    return DV_UNDETERMINED;
  }
  if (len < (size_t)window_bits) {
    return DV_UNDETERMINED;
  }
  const long min_windows = dv_min_recovery_windows(n, window_bits);
  const long windows = (long)((len - (size_t)window_bits) / (size_t)n) + 1;
  if (windows < min_windows) {
    return DV_UNDETERMINED;
  }

  uint32_t basis[DV_HARD_WINDOW_CAP];
  double self_satisfaction = 0.0;
  const int dimension =
      recover_basis(sample, len, n, window_bits, basis, &self_satisfaction);
  if (dimension < 0) {
    return DV_UNDETERMINED; /* allocation failure */
  }
  if (dimension == 0) {
    return 0.0; /* no self-validating dual structure -> no code present */
  }
  /* Map the self cross-satisfaction (~1 clean, the floor when barely detected)
   * onto a probability - the single-stream analog of dv_compare's rescale. */
  return dv_clamp01(2.0 * (self_satisfaction - 0.5));
}
