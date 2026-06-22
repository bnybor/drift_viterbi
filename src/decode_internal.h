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

#ifndef DRIFT_VITERBI_DECODE_INTERNAL_H
#define DRIFT_VITERBI_DECODE_INTERNAL_H

/*
 * Private decoder internals shared by the single-stream decoder (decode.c) and
 * the multi-decoder (multi.c). Not installed.
 *
 * The sliding-window Viterbi state splits in two: a `dv_decode_ctx` holds
 * everything identical across codes decoded over the same stream - the received
 * buffer, the cadence (read_base/steps/decided and the shared re-anchor history),
 * the channel-derived cost constants, and the trellis dimensions - while a
 * `dv_trellis` holds one code's per-step working state. The single decoder is one
 * ctx + one trellis; the multi-decoder is one ctx + an array of trellises stepped
 * in lockstep, so a single received buffer and a single re-anchor decision serve
 * them all.
 */

#include <drift_viterbi/decode.h> /* dv_stream_params, dv_code (opaque), uint8_t */

#include <stddef.h>
#include <stdint.h>

/*
 * Opt-in internal assertions. The core is built -ffreestanding -fno-builtin
 * -nostdlib, so it cannot use <assert.h> (that would pull in __assert_fail /
 * abort from libc). __builtin_trap is a compiler intrinsic - not a libc symbol
 * and not suppressed by -fno-builtin - so it traps without breaking the
 * freestanding link. Off by default (zero cost); define DV_DEBUG_ASSERT to arm
 * the load-bearing invariants in the decoder (see dv_trellis_trace). */
#if defined(DV_DEBUG_ASSERT)
#define DV_ASSERT(cond) \
  do {                  \
    if (!(cond)) {      \
      __builtin_trap(); \
    }                   \
  } while (0)
#else
#define DV_ASSERT(cond) ((void)0)
#endif

/* Backpointer for one node: where it came from (prev_state, prev_drift_index)
 * and the input bit that got there. */
typedef struct {
  int prev_state;
  int prev_drift_index;
  unsigned char bit;
} dv_backpointer;

/* Shared decode context: received buffer, cadence, channel constants, and
 * trellis dimensions - all identical across codes decoded together. */
typedef struct {
  int n, max_drift, num_states, drift_width, decision_depth;
  /* Branch-metric constants, in cost (negative-log-likelihood) units. */
  double cost_match, cost_miss, cost_erase, cost_keep, cost_ins, cost_del;
  /* Lock detection reference costs (channel-derived, code-independent). */
  double expected_lock, expected_unlock;

  int *shift; /* [decision_depth] shared re-anchor sigma per step */

  long long steps;   /* trellis steps processed                      */
  long long decided; /* decisions emitted (next step index to emit)  */

  /* Received-bit buffer: valid bits live in received[0 .. received_length).
   * read_base is the buffer index of the current step's zero-drift read base. */
  uint8_t *received;
  int received_capacity, received_length, read_base;
} dv_decode_ctx;

/* One code's trellis working state. */
typedef struct {
  const dv_code *code;          /* borrowed                                  */
  double *metric;               /* [num_states*drift_width] node costs       */
  double *next_metric;          /* [num_states*drift_width] scratch          */
  dv_backpointer *backpointers; /* [decision_depth*num_states*drift_width]   */
  double *alignment;            /* [(n+1)*(n+2*max_drift+1)] DP scratch      */
  double smoothed_cost;         /* EWMA of best-path per-step cost           */
} dv_trellis;

/* Validate `params`, take dimensions from `code`, allocate the shared buffers,
 * and zero the cadence. Returns DV_OK or a negative DV_ERR_*. A ctx that failed
 * (or was zero-initialised) is safe to pass to dv_decode_ctx_free. */
int dv_decode_ctx_init(dv_decode_ctx *ctx, const dv_stream_params *params,
                       const dv_code *code);
void dv_decode_ctx_free(dv_decode_ctx *ctx);

/* Allocate one trellis's per-code arrays (sized from `ctx`) and initialise its
 * metric and smoothed cost for blind acquisition. Returns DV_OK or negative; a
 * failed/zeroed trellis is safe to dv_trellis_free. */
int dv_trellis_init(dv_trellis *tr, const dv_decode_ctx *ctx,
                    const dv_code *code);
void dv_trellis_free(dv_trellis *tr);

/* Append `n_in` received bits to the shared buffer. Returns DV_OK or negative. */
int dv_decode_feed(dv_decode_ctx *ctx, const uint8_t *in, int n_in);

/* Advance every trellis one fused step: one re-anchor sigma (picked from the
 * best-fitting trellis) is applied to all, then each runs its forward pass, then
 * the shared cadence advances. With n == 1 this is the plain single-decoder step. */
void dv_decode_step(dv_decode_ctx *ctx, dv_trellis *trs, size_t n);

/* Lowest-cost node at a trellis's current frontier. */
int dv_trellis_frontier(const dv_decode_ctx *ctx, const dv_trellis *tr);

/* Input bit a trellis decided at step `target`, traced from node `frontier`. */
unsigned char dv_trellis_trace(const dv_decode_ctx *ctx, const dv_trellis *tr,
                               int frontier, long long target);

/* Probability the trellis is locked onto this code's stream (0..1). */
double dv_trellis_lock(const dv_decode_ctx *ctx, const dv_trellis *tr);

#endif /* DRIFT_VITERBI_DECODE_INTERNAL_H */
