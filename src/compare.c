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

/* ========================================================================== *
 *  Graded blind detection (dv_detect)                                        *
 *                                                                            *
 *  dv_detect answers whether a buffer carries a rate-1/n, constraint-        *
 *  length-k code without knowing the generators or the sent bits, and keeps  *
 *  that answer graded, sliding smoothly to zero as noise rises. It rests on  *
 *  four parts:                                                               *
 *                                                                            *
 *   Part 1 - graded statistic. The WHT coefficient of a parity vector v is   *
 *     spectrum[v] = (#satisfied - #violated) windows; its alignment          *
 *     a(v) = spectrum[v] / window_count = 2*sat_frac - 1 is in [-1, 1].      *
 *     Under H0 (random data) a(v) ~ N(0, 1/window_count), so the largest     *
 *     of the ~2^window_bits vectors sits at roughly                          *
 *     sqrt(2*window_bits*ln2)/sqrt(window_count) (an extreme-value floor).   *
 *     The strongest independent peaks are summed in excess of that floor:    *
 *     a clean code saturates the sum, noise slides it smoothly to zero.      *
 *                                                                            *
 *   Part 2 - pool evidence across the whole stream, per phase. The WHT is    *
 *     linear in the histogram, so every window of the stream is              *
 *     histogrammed. Because an indel flips the slide-by-n phase, n separate  *
 *     phase histograms are kept (start position mod n) and the strongest     *
 *     phase is taken: each clean run between indels lands wholly in one      *
 *     phase bin, raising window_count (hence SNR) and surviving scattered    *
 *     indels.                                                                *
 *                                                                            *
 *   Part 3 - drift-corrected re-accumulation. Given a candidate basis, the   *
 *     stream is walked re-anchoring the frame by +/-1 bit per symbol to      *
 *     keep the basis satisfied (a greedy analog of the decoder's drift       *
 *     window), and re-scored along that path, pulling post-indel windows     *
 *     back into alignment so they reinforce the peaks.                       *
 *                                                                            *
 *   Part 4 - a sequential detector (dv_detect_sequential): recover a basis   *
 *     from a warmup prefix, then run a one-sided CUSUM of the per-window     *
 *     satisfied-check log-likelihood ratio (H1 vs the H0 rate 0.5), so a     *
 *     "code present" call fires as soon as enough evidence accrues rather    *
 *     than waiting for a whole-buffer verdict - graceful and low-latency.    *
 *                                                                            *
 *  Detection SNR is fundamentally sqrt(window_count)-limited: the floor      *
 *  shrinks only as 1/sqrt(window_count), so a longer capture detects a       *
 *  noisier stream. That is physics, not a tunable.                           *
 * ========================================================================== */

/* sqrt built from the stdlib's exp/log (the freestanding build has no libm
 * sqrt). Exact enough for a noise-floor estimate. */
static double dv_sqrt(double x) { return x > 0.0 ? dv_exp(0.5 * dv_log(x)) : 0.0; }

/* Safety margin (in units of the H0 standard deviation) added to the extreme-
 * value peak floor, so random data reads ~0 rather than skimming the floor. */
static const double DV_DETECT_PEAK_MARGIN = 1.0;

/* H0 extreme-value floor on |alignment|: the expected largest of ~2^window_bits
 * unit-Gaussian WHT coefficients is ~sqrt(2*window_bits*ln2) standard
 * deviations, and one std of alignment is 1/sqrt(window_count). A selected peak
 * must clear this to count as structure rather than the luckiest noise bin. */
static double dv_peak_floor(int window_bits, long window_count) {
  if (window_count <= 0) return 1.0;
  const double sigmas =
      dv_sqrt(2.0 * (double)window_bits * 0.69314718055994530942) +
      DV_DETECT_PEAK_MARGIN;
  return sigmas / dv_sqrt((double)window_count);
}

/* Turn a basis's summed excess alignment into a [0,1] confidence. The sum
 * saturates: a clean code contributes ~dimension * 1, noise contributes ~0. */
static double dv_detect_confidence(double excess_sum) {
  return dv_clamp01(excess_sum);
}

/*
 * Degeneracy guard. A genuine rate-1/n, constraint-k code carries exactly
 * (k+1) information bits in an n*(k+1)-bit window (one per input step it
 * spans), so its dual (parity-check) space there has dimension at most
 * (n-1)*(k+1). A degenerate stream - a constant, a low-period pattern, or a
 * strongly DC-biased one - instead has its dual fill nearly the whole window:
 * every parity vector sits at alignment +/-1, so the energy statistic
 * saturates and a pure energy detector calls it a code. Heavy erasure is the
 * common trap: DV_ERASURE (0xFF) reads as a 1 bit, so as the erasure rate
 * climbs the stream tends to all-ones, and even at moderate rates the 1-bias
 * lights up every single-bit check. Recovered duals wider than this cap are
 * rejected. window_bits = n*(k+1), so (k+1) = window_bits / n. */
static int dv_max_dual(int n, int window_bits) {
  return (n - 1) * (window_bits / n);
}

/*
 * Lever 1 scorer over a WHT spectrum: greedily select the strongest linearly
 * independent parity vectors (largest |alignment|), with no hard satisfaction
 * gate, and accumulate each one's alignment in excess of the H0 peak floor.
 * Independence is enforced by GF(2) reduction so a check and its multiples are
 * not double-counted. Also returns the selected basis (for lever 3 / the
 * sequential path) via basis_out/dim_out when non-NULL. Rejects (excess 0,
 * dimension 0) when the recovered dual is too wide to be a genuine rate-1/n
 * code - the degeneracy guard above.
 */
static double dv_spectral_excess(const int *spectrum, long window_count, int n,
                                 int window_bits, uint32_t *basis_out,
                                 int *dim_out) {
  const double floor = dv_peak_floor(window_bits, window_count);
  const size_t bin_count = (size_t)1 << window_bits;
  uint32_t row_for_bit[DV_MAX_WINDOW_BITS];
  dv_memset(row_for_bit, 0, sizeof(row_for_bit));

  double excess_sum = 0.0;
  int dimension = 0;
  /* At most window_bits independent directions exist; pull them out strongest
   * first by repeated max-scan. Each scan skips vectors already in the span. */
  for (int picked = 0; picked < window_bits; ++picked) {
    double best_align = floor; /* must beat the floor to be picked at all */
    uint32_t best_vec = 0;
    for (size_t v = 1; v < bin_count; ++v) {
      double align = (double)spectrum[v] / (double)window_count;
      if (align < 0.0) align = -align;
      if (align <= best_align) continue;
      /* independence test: reduce v against the current selected span */
      uint32_t row = (uint32_t)v;
      for (int bit = window_bits - 1; bit >= 0; --bit) {
        if (!((row >> bit) & 1u)) continue;
        if (row_for_bit[bit]) row ^= row_for_bit[bit];
        else break;
      }
      if (row == 0) continue; /* already spanned */
      best_align = align;
      best_vec = (uint32_t)v;
    }
    if (best_vec == 0) break; /* nothing left above the floor */
    /* commit best_vec to the span */
    uint32_t row = best_vec;
    for (int bit = window_bits - 1; bit >= 0; --bit) {
      if (!((row >> bit) & 1u)) continue;
      if (row_for_bit[bit]) row ^= row_for_bit[bit];
      else { row_for_bit[bit] = row; break; }
    }
    if (basis_out) basis_out[dimension] = best_vec;
    ++dimension;
    excess_sum += best_align - floor;
  }
  /* Degeneracy guard: a dual this wide is a constant/biased stream, not a
   * code. Reject rather than let the saturated energy through. */
  if (dimension > dv_max_dual(n, window_bits)) {
    if (dim_out) *dim_out = 0;
    return 0.0;
  }
  if (dim_out) *dim_out = dimension;
  return excess_sum;
}

/*
 * Lever 2 recovery for one phase: pool the WHOLE stream (from byte `phase`,
 * sliding by n) into one histogram, transform, and score with lever 1. Reuses
 * dv_dual_spectrum, capped at DV_MAX_CROSS_WINDOWS. WHT path only (window_bits
 * <= DV_MAX_WINDOW_BITS); wider codes use dv_basis_excess_wide below.
 */
static double dv_phase_excess_wht(const uint8_t *stream, size_t len, int n,
                                  int window_bits, int phase,
                                  uint32_t *basis_out, int *dim_out) {
  if ((size_t)phase + (size_t)window_bits > len) {
    if (dim_out) *dim_out = 0;
    return 0.0;
  }
  long used = 0;
  int *spectrum =
      dv_dual_spectrum(stream + phase, len - (size_t)phase, n, window_bits,
                       DV_MAX_CROSS_WINDOWS, &used);
  if (!spectrum) {
    if (dim_out) *dim_out = -1; /* allocation failure */
    return 0.0;
  }
  double excess =
      dv_spectral_excess(spectrum, used, n, window_bits, basis_out, dim_out);
  dv_free(spectrum);
  return excess;
}

/*
 * Graded score of an explicit basis at a fixed phase: each check's satisfied
 * fraction over the whole stream, summed in excess of the H0 floor. The wide-
 * window (no-WHT) analog of dv_spectral_excess; also reused by lever 3.
 */
static double dv_basis_excess_fixed(const uint8_t *stream, size_t len, int n,
                                    int window_bits, int phase,
                                    const uint32_t *basis, int dimension) {
  if (dimension <= 0) return 0.0;
  long sat[DV_HARD_WINDOW_CAP];
  dv_memset(sat, 0, sizeof(sat));
  long wc = 0;
  for (long pos = phase; pos + window_bits <= (long)len; pos += n) {
    uint32_t packed = dv_window(stream, (size_t)pos, window_bits);
    for (int i = 0; i < dimension; ++i)
      if (!__builtin_parity(packed & basis[i])) ++sat[i];
    ++wc;
    if (wc >= DV_MAX_CROSS_WINDOWS) break;
  }
  if (wc == 0) return 0.0;
  const double floor = dv_peak_floor(window_bits, wc);
  double excess = 0.0;
  for (int i = 0; i < dimension; ++i) {
    double align = 2.0 * ((double)sat[i] / (double)wc) - 1.0;
    if (align < 0.0) align = -align;
    if (align > floor) excess += align - floor;
  }
  return excess;
}

/*
 * Lever 3: re-score `basis` along a drift-corrected path. Starting at `phase`,
 * advance one symbol at a time but let the frame slip by -1/0/+1 bit per step
 * (bounded to +/-DV_MAX_CUMULATIVE_DRIFT net), greedily choosing the slip that
 * best satisfies the basis. Re-anchoring this way pulls post-indel windows back
 * into frame so they reinforce the peaks instead of smearing them.
 */
static double dv_basis_excess_drift(const uint8_t *stream, size_t len, int n,
                                    int window_bits, int phase,
                                    const uint32_t *basis, int dimension) {
  if (dimension <= 0) return 0.0;
  long sat[DV_HARD_WINDOW_CAP];
  dv_memset(sat, 0, sizeof(sat));
  long wc = 0;
  long pos = phase;
  long net_drift = 0; /* signed bits of accumulated slip from the nominal frame */
  /* visit candidate slips in the order {0, +1, -1} so equal-scoring ties keep
   * step 0 (assume no indel unless a slip strictly helps). */
  static const int kSteps[3] = {0, 1, -1};
  while (pos + window_bits <= (long)len) {
    uint32_t packed = dv_window(stream, (size_t)pos, window_bits);
    for (int i = 0; i < dimension; ++i)
      if (!__builtin_parity(packed & basis[i])) ++sat[i];
    ++wc;
    if (wc >= DV_MAX_CROSS_WINDOWS) break;
    /* pick the next frame pos + n + step that best satisfies the basis. */
    long best_pos = -1;
    int best_step = 0;
    int best_good = -1;
    for (int si = 0; si < 3; ++si) {
      const int s = kSteps[si];
      if (dv_abs((int)(net_drift + s)) > DV_MAX_CUMULATIVE_DRIFT) continue;
      const long cand = pos + n + s;
      if (cand < 0 || cand + window_bits > (long)len) continue;
      uint32_t cw = dv_window(stream, (size_t)cand, window_bits);
      int good = 0;
      for (int i = 0; i < dimension; ++i)
        if (!__builtin_parity(cw & basis[i])) ++good;
      if (good > best_good) {
        best_good = good;
        best_pos = cand;
        best_step = s;
      }
    }
    if (best_pos < 0) break;
    net_drift += best_step;
    pos = best_pos;
  }
  if (wc == 0) return 0.0;
  const double floor = dv_peak_floor(window_bits, wc);
  double excess = 0.0;
  for (int i = 0; i < dimension; ++i) {
    double align = 2.0 * ((double)sat[i] / (double)wc) - 1.0;
    if (align < 0.0) align = -align;
    if (align > floor) excess += align - floor;
  }
  return excess;
}

/* Recover a candidate basis for one phase (wide-window codes, no WHT). The
 * nullspace recovery is exact but brittle: a single corrupted window in the
 * recovery segment (an indel or erasure) yields a useless basis. So we scan
 * SEGMENTS forward - as the original recover_basis does - and keep the one
 * whose basis best explains the whole stream under the graded score; a clean
 * run between indels validates itself, and the drift pass then carries it
 * across the rest. */
static double dv_phase_excess_wide(const uint8_t *stream, size_t len, int n,
                                   int window_bits, int phase) {
  if ((size_t)phase + (size_t)window_bits > len) return 0.0;
  const long seg = 4L * window_bits;
  const long total_windows =
      (long)((len - (size_t)phase - (size_t)window_bits) / (size_t)n) + 1;
  const long min_windows = dv_min_recovery_windows(n, window_bits);

  uint32_t best_basis[DV_HARD_WINDOW_CAP];
  int best_dim = 0;
  double best_fixed = 0.0;
  const int max_dual = dv_max_dual(n, window_bits);
  for (long start = 0; start + min_windows <= total_windows; start += seg) {
    const size_t off = (size_t)phase + (size_t)start * (size_t)n;
    uint32_t cand[DV_HARD_WINDOW_CAP];
    int dim = recover_segment_nullspace(stream + off, len - off, n, window_bits,
                                        seg, cand);
    if (dim <= 0 || dim > max_dual) continue; /* skip empty/degenerate duals */
    double fixed = dv_basis_excess_fixed(stream, len, n, window_bits, phase,
                                         cand, dim);
    if (fixed > best_fixed) {
      best_fixed = fixed;
      best_dim = dim;
      dv_memcpy(best_basis, cand, (size_t)dim * sizeof(*cand));
    }
  }
  if (best_dim <= 0) return 0.0;
  /* Lever 3: carry the best basis across the stream along a drift-corrected
   * path and keep whichever score is stronger. */
  double drift = dv_basis_excess_drift(stream, len, n, window_bits, phase,
                                       best_basis, best_dim);
  return best_fixed > drift ? best_fixed : drift;
}

/*
 * Detect whether a single buffer carries any rate-1/n, constraint-length-k
 * convolutional code, without identifying which one. Blindly recovers the
 * stream's dual (parity-check) space and measures how strongly the stream's
 * windows concentrate on it, relative to the H0 noise floor: a coded stream
 * scores ~1, random data ~0, and the fall-off between is GRADED rather than a
 * cliff (see the lever notes above). Tolerates substitution, erasure, and -
 * via per-phase pooling (lever 2) and drift-corrected re-accumulation (lever
 * 3) - insertion/deletion drift.
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

  double best_excess = 0.0;
  /* Lever 2: try every framing phase; an indel between clean runs routes them
   * to different phase bins, so the best phase pools the most clean evidence. */
  for (int phase = 0; phase < n; ++phase) {
    double excess;
    if (window_bits <= DV_MAX_WINDOW_BITS) {
      uint32_t basis[DV_HARD_WINDOW_CAP];
      int dim = 0;
      excess = dv_phase_excess_wht(sample, len, n, window_bits, phase, basis,
                                   &dim);
      if (dim < 0) {
        return DV_UNDETERMINED; /* allocation failure */
      }
      /* Lever 3: re-score the recovered basis along a drift-corrected path and
       * keep whichever explains the stream better. */
      if (dim > 0) {
        double drift = dv_basis_excess_drift(sample, len, n, window_bits, phase,
                                             basis, dim);
        if (drift > excess) excess = drift;
      }
    } else {
      excess = dv_phase_excess_wide(sample, len, n, window_bits, phase);
    }
    if (excess > best_excess) best_excess = excess;
  }
  return dv_detect_confidence(best_excess);
}

/* -- sequential (streaming) detection: lever 4 ----------------------------- */

/* Nominal H1 per-check satisfaction used by the CUSUM log-likelihood ratio:
 * any value comfortably above the H0 rate 0.5 works; 0.75 fires when windows
 * consistently satisfy more than ~60% of the recovered checks. */
static const double DV_DETECT_SEQ_Q1 = 0.75;

/* SPRT error targets -> CUSUM decision threshold log((1-beta)/alpha). */
static const double DV_DETECT_SEQ_ALPHA = 1e-3;
static const double DV_DETECT_SEQ_BETA = 1e-3;

/* Warmup prefix (in symbols) used to recover the steering basis before the
 * CUSUM begins; capped to the available stream. */
#define DV_DETECT_SEQ_WARMUP_WINDOWS 256

/*
 * Sequential blind detection. Recovers a steering basis from a warmup prefix
 * (best phase, same graded recovery as dv_detect), then runs a one-sided CUSUM
 * of the per-window satisfied-check log-likelihood ratio. Returns the bit index
 * at which "code present" is first declared, or -1 if the stream never crosses
 * the threshold (or no basis could be recovered). When confidence_out is
 * non-NULL it receives a [0,1] reading: 1.0 once declared, else the CUSUM's
 * closest approach to the threshold as a fraction.
 */
long dv_detect_sequential(int n, int k, const uint8_t *sample, size_t len,
                          double *confidence_out) {
  if (confidence_out) *confidence_out = 0.0;
  if (!sample) return -1;
  const int window_bits = dv_window_bits(n, k);
  if (window_bits == 0) return -1;
  if (len < (size_t)window_bits) return -1;
  const long min_windows = dv_min_recovery_windows(n, window_bits);
  const long windows = (long)((len - (size_t)window_bits) / (size_t)n) + 1;
  if (windows < min_windows) return -1;

  /* Warmup recovery: pick the phase whose prefix exposes the strongest dual. */
  size_t warmup = (size_t)DV_DETECT_SEQ_WARMUP_WINDOWS * (size_t)n +
                  (size_t)window_bits;
  if (warmup > len) warmup = len;
  uint32_t basis[DV_HARD_WINDOW_CAP];
  int dimension = 0;
  int best_phase = 0;
  double best_excess = 0.0;
  for (int phase = 0; phase < n; ++phase) {
    uint32_t cand[DV_HARD_WINDOW_CAP];
    int dim = 0;
    double excess;
    if (window_bits <= DV_MAX_WINDOW_BITS) {
      excess = dv_phase_excess_wht(sample, warmup, n, window_bits, phase, cand,
                                   &dim);
      if (dim < 0) return -1;
    } else {
      if ((size_t)phase + (size_t)window_bits > warmup) continue;
      dim = recover_segment_nullspace(sample + phase, warmup - (size_t)phase, n,
                                      window_bits, 4L * window_bits, cand);
      if (dim > dv_max_dual(n, window_bits)) dim = 0; /* degenerate: reject */
      excess = dv_basis_excess_fixed(sample, warmup, n, window_bits, phase, cand,
                                     dim);
    }
    if (dim > 0 && excess > best_excess) {
      best_excess = excess;
      best_phase = phase;
      dimension = dim;
      dv_memcpy(basis, cand, (size_t)dim * sizeof(*cand));
    }
  }
  if (dimension <= 0) return -1; /* no structure to steer the CUSUM */

  const double l_sat = dv_log(DV_DETECT_SEQ_Q1 / 0.5);
  const double l_unsat = dv_log((1.0 - DV_DETECT_SEQ_Q1) / 0.5);
  const double threshold =
      dv_log((1.0 - DV_DETECT_SEQ_BETA) / DV_DETECT_SEQ_ALPHA);

  double cusum = 0.0;
  double closest = 0.0;
  for (long pos = best_phase; pos + window_bits <= (long)len; pos += n) {
    const int good =
        dv_window_good(sample, pos, window_bits, basis, dimension);
    const double llr =
        (double)good * l_sat + (double)(dimension - good) * l_unsat;
    cusum += llr;
    if (cusum < 0.0) cusum = 0.0;
    if (cusum > closest) closest = cusum;
    if (cusum >= threshold) {
      if (confidence_out) *confidence_out = 1.0;
      return pos; /* bit index of the declaration */
    }
  }
  if (confidence_out) *confidence_out = dv_clamp01(closest / threshold);
  return -1;
}
