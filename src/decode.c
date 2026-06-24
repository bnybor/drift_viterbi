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

#include <drift_viterbi/decode.h>

#include <drift_viterbi/stdlib.h>

#include <math.h>

#include "decode_internal.h"
#include "dv_internal.h"

/* ------------------------------------------------------------------------- */
/* Streaming (sliding-window) decoder                                        */
/* ------------------------------------------------------------------------- */

/* clang-format off */
/*
 * Runs the forward add-compare-select pass continuously over a buffer of
 * received bits, keeping only `decision_depth` steps of backpointers. A bit is
 * committed decision_depth steps after it is first seen, by which point the
 * candidate paths have merged. Output emerges at fixed latency decision_depth
 * with bounded memory and no frame boundaries.
 *
 * Each step emits one input bit's group of n coded bits, scored by a per-edge
 * bit-level alignment (see align_fill) that may insert or delete individual
 * received bits anywhere within the group. A node's drift is its running
 * net (insertions - deletions); a branch that consumes n + Delta received bits
 * moves drift by Delta, so indels at arbitrary bit positions - not just group
 * boundaries - are tracked exactly.
 *
 * Re-anchoring keeps the drift window centred on the committed timing: each
 * step the window may be shifted by sigma in {-1,0,+1} (folded into the read
 * cursor), so the net cumulative drift can grow without bound while each node's
 * stored drift stays inside +/- max_drift. The shift is recorded per step so
 * traceback can translate node indices across the moving coordinate frame.
 *
 * The state splits into a shared dv_decode_ctx (received buffer, cadence, cost
 * constants, dimensions) and a per-code dv_trellis (metric/backpointers/etc).
 * Every step function takes (ctx, trellis); the shared dv_decode_step() advances
 * one or many trellises in lockstep over the one buffer with one re-anchor.
 */
/* clang-format on */

/* Flat index into the [num_states][drift_width] metric and backpointer arrays
 * for the node at encoder state `state` and drift index `drift_index`
 * (0..drift_width-1; drift_index == max_drift means zero drift). */
static size_t node_at(int state, int drift_index, int drift_width) {
  return (size_t)state * drift_width + drift_index;
}

/* -- received buffer ------------------------------------------------------- */

/* Drop the dead prefix. Keep 2*max_drift bits of history below read_base so a
 * re-anchor that steps the cursor back still has its window buffered. */
static void compact_received(dv_decode_ctx *ctx) {
  int keep_from = ctx->read_base - 2 * ctx->max_drift;
  if (keep_from <= 0) {
    return;
  }
  dv_memmove(ctx->received, ctx->received + keep_from,
          (size_t)(ctx->received_length - keep_from));
  ctx->received_length -= keep_from;
  ctx->read_base -= keep_from;
}

/* Ensure room for `extra` more bits, compacting then growing as needed. */
static int reserve_received(dv_decode_ctx *ctx, int extra) {
  compact_received(ctx);
  if (ctx->received_length + extra > ctx->received_capacity) {
    int new_capacity = ctx->received_capacity * 2;
    if (new_capacity < ctx->received_length + extra) {
      new_capacity = ctx->received_length + extra;
    }
    uint8_t *new_buffer = dv_realloc(ctx->received, (size_t)new_capacity);
    if (!new_buffer) {
      return DV_ERR_ALLOC;
    }
    ctx->received = new_buffer;
    ctx->received_capacity = new_capacity;
  }
  return DV_OK;
}

/* -- core trellis ---------------------------------------------------------- */

/* Flat index of the lowest-cost node at the current frontier (the node states
 * one step past the last one processed). */
int dv_trellis_frontier(const dv_decode_ctx *ctx, const dv_trellis *tr) {
  const size_t count = (size_t)ctx->num_states * ctx->drift_width;
  double best_cost = INFINITY;
  int best = 0;
  for (size_t i = 0; i < count; ++i) {
    if (tr->metric[i] < best_cost) {
      best_cost = tr->metric[i];
      best = (int)i;
    }
  }
  return best;
}

/* Choose this step's re-anchor shift: nudge the window one step toward centre
 * when the best node's drift leaves a deadband, so its drift stays well inside
 * the window even as cumulative drift grows. */
static int pick_shift(const dv_decode_ctx *ctx, const dv_trellis *tr) {
  if (ctx->max_drift == 0) {
    return 0; /* no drift tracking: window is one wide, nothing to shift */
  }
  const int best_drift_index = dv_trellis_frontier(ctx, tr) % ctx->drift_width;
  const int drift = best_drift_index - ctx->max_drift;
  const int deadband = (ctx->max_drift + 1) / 2;
  if (drift >= deadband) {
    return +1;
  }
  if (drift <= -deadband) {
    return -1;
  }
  return 0;
}

/* clang-format off */
/* Fill the alignment DP for one edge: align the `n` expected output bits against
 * the received bits starting at buffer index `base`, allowing per-bit insertion,
 * deletion, and substitution. cost_table[j][consumed] (stored row-major at
 * j*(max_consume+1)+consumed) is the min cost to align the first j expected bits
 * while consuming `consumed` received bits; max_consume = n + 2*max_drift is the
 * most bits any branch can consume. Reads outside the buffered region
 * (buffer_index < 0 or >= received_length) are infeasible, so only the deletion
 * move is available there - this is what keeps the ends of the stream safe.
 *
 * The branch cost into a node at ending drift di' is then
 * cost_table[n][n + (di' - di)]: consuming n + Delta received bits to emit the
 * group shifts drift by Delta. A matched/substituted bit also pays cost_keep (the
 * per-bit "not an indel" cost), so the model is fully bit-level rather than
 * per-group. */
/* clang-format on */
static void align_fill(const dv_decode_ctx *ctx, dv_trellis *tr,
                       const uint8_t *expected, int base) {
  const int n = ctx->n, max_consume = ctx->n + 2 * ctx->max_drift,
            stride = max_consume + 1;
  /* The per-bit match cost and in-range gate for each received position are
   * precomputed once per step (fill_match_costs) and shared by every align_fill;
   * `off` maps this edge's read base into that table. The arithmetic below is
   * identical to deriving them per cell. */
  const int off = base - tr->match_lo;
  const signed char *in_range = tr->in_range;
  const double *match_cost0 = tr->match_cost0, *match_cost1 = tr->match_cost1;
  const double cost_ins = ctx->cost_ins, cost_del = ctx->cost_del;
  double *cost_table = tr->alignment;

  cost_table[0] = 0.0;
  for (int consumed = 1; consumed <= max_consume; ++consumed) {
    cost_table[consumed] = in_range[off + consumed - 1]
                               ? cost_table[consumed - 1] + cost_ins
                               : INFINITY;
  }
  for (int j = 1; j <= n; ++j) {
    double *cost_row = cost_table + (size_t)j * stride;
    const double *prev_row = cost_table + (size_t)(j - 1) * stride;
    const double *match_cost = expected[j - 1] ? match_cost1 : match_cost0;
    cost_row[0] =
        prev_row[0] + cost_del; /* delete expected[j-1], consume nothing */
    for (int consumed = 1; consumed <= max_consume; ++consumed) {
      double best = prev_row[consumed] + cost_del; /* deletion always avail. */
      const int position = off + consumed - 1;
      if (in_range[position]) {
        const double align_cost =
            prev_row[consumed - 1] + match_cost[position]; /* match / sub */
        const double insert_cost =
            cost_row[consumed - 1] + cost_ins; /* extra received bit */
        if (align_cost < best) best = align_cost;
        if (insert_cost < best) best = insert_cost;
      }
      cost_row[consumed] = best;
    }
  }
}

/* Precompute the step's received-window match costs once, shared by every
 * align_fill this step (they all read the same buffer with the same channel
 * cost constants). Position p maps to absolute received index match_lo + p;
 * match_cost{0,1}[p] is the cost of aligning an expected 0/1 bit there (an erased
 * bit costs cost_erase either way), and in_range[p] gates positions off the ends
 * of the buffer (where only deletion is available - see align_fill). The window
 * spans every (source drift, consumed) an edge can touch: n + 4*max_drift wide. */
static void fill_match_costs(dv_decode_ctx *ctx, dv_trellis *tr) {
  const int window = ctx->n + 4 * ctx->max_drift;
  const double keep = ctx->cost_keep, match = ctx->cost_match,
               miss = ctx->cost_miss, erase = ctx->cost_erase;
  tr->match_lo = ctx->read_base - ctx->max_drift;
  for (int p = 0; p < window; ++p) {
    const int absolute = tr->match_lo + p;
    if (absolute >= 0 && absolute < ctx->received_length) {
      const uint8_t received_bit = ctx->received[absolute];
      if (received_bit == DV_ERASURE) {
        tr->match_cost0[p] = keep + erase;
        tr->match_cost1[p] = keep + erase;
      } else {
        tr->match_cost0[p] = keep + (received_bit == 0 ? match : miss);
        tr->match_cost1[p] = keep + (received_bit == 1 ? match : miss);
      }
      tr->in_range[p] = 1;
    } else {
      tr->in_range[p] = 0;
    }
  }
}

/* Subtract the lowest node cost from every node, so the best one sits at 0.
 * Over an unbounded stream this keeps the costs from growing without limit.
 * Returns the amount subtracted: the best path's cost increment this step. */
static double normalize(double *metric, size_t count) {
  double lowest = INFINITY;
  for (size_t i = 0; i < count; ++i) {
    if (metric[i] < lowest) {
      lowest = metric[i];
    }
  }
  if (lowest == INFINITY) {
    return 0.0;
  }
  if (lowest > 0.0) {
    for (size_t i = 0; i < count; ++i) {
      if (metric[i] != INFINITY) {
        metric[i] -= lowest;
      }
    }
  }
  return lowest;
}

/* Shift one trellis's drift window by `sigma` (each node's drift index
 * drift_index -> drift_index - sigma); the read cursor is advanced once, by
 * dv_decode_step, to match. Nodes shifted out of the window are dropped. Uses
 * next_metric as scratch.
 *
 * The dropped edge slot is set to INFINITY (sigma > 0 empties the top slot,
 * sigma < 0 the bottom). forward_pass then skips those slots as sources, so the
 * emptied edge can never be recorded as a backpointer's prev_drift_index - which
 * is exactly what keeps dv_trellis_trace's `prev_drift_index + sigma` inside
 * [0, drift_width) when it steps back across this re-anchor. */
static void reanchor_metric(const dv_decode_ctx *ctx, dv_trellis *tr,
                            int sigma) {
  const int num_states = ctx->num_states, drift_width = ctx->drift_width;
  for (int state = 0; state < num_states; ++state) {
    for (int drift_index = 0; drift_index < drift_width; ++drift_index) {
      const int source = drift_index + sigma;
      tr->next_metric[node_at(state, drift_index, drift_width)] =
          (source >= 0 && source < drift_width)
              ? tr->metric[node_at(state, source, drift_width)]
              : INFINITY;
    }
  }
  dv_memcpy(tr->metric, tr->next_metric,
         (size_t)num_states * drift_width * sizeof(double));
}

/* One trellis's forward pass for the current step: run the add-compare-select
 * from tr->metric into tr->next_metric (recording backpointers in this step's
 * layer), normalise, update the smoothed cost, and swap. Reads ctx->read_base
 * (already re-anchored for this step) and the shared received buffer. Branches
 * that would read outside the buffered window are skipped. */
static void forward_pass(dv_decode_ctx *ctx, dv_trellis *tr) {
  const dv_code *code = tr->code;
  const int n = ctx->n, max_drift = ctx->max_drift, num_states = ctx->num_states,
            drift_width = ctx->drift_width;
  const size_t count = (size_t)num_states * drift_width;
  const int max_consume =
      ctx->n + 2 * ctx->max_drift; /* most received bits a branch can consume */
  const int stride = max_consume + 1;

  for (size_t i = 0; i < count; ++i) {
    tr->next_metric[i] = INFINITY;
  }
  dv_backpointer *layer =
      tr->backpointers + (size_t)(ctx->steps % ctx->decision_depth) * count;

  /* An edge's alignment DP depends only on its output group and source drift, so
   * compute the final row once per (group, source drift) - at most 2^n groups
   * instead of 2*num_states (state, bit) pairs - and let the scatter look it up.
   * Source drifts with no live node are skipped: the scatter never reads their
   * rows (it skips dead nodes), so this is exact and also cuts acquisition. */
  fill_match_costs(ctx, tr);
  for (int drift_index = 0; drift_index < drift_width; ++drift_index) {
    int live = 0;
    for (int state = 0; state < num_states; ++state) {
      if (tr->metric[(size_t)state * drift_width + drift_index] != INFINITY) {
        live = 1;
        break;
      }
    }
    if (!live) {
      continue;
    }
    const int base = ctx->read_base + (drift_index - max_drift);
    for (int g = 0; g < tr->n_groups; ++g) {
      align_fill(ctx, tr, tr->group_expected[g], base);
      dv_memcpy(
          tr->align_final + ((size_t)g * drift_width + drift_index) * stride,
          tr->alignment + (size_t)n * stride, (size_t)stride * sizeof(double));
    }
  }

  /* Scatter. The node/edge iteration order (state, drift, bit, ending drift) is
   * the same as the per-edge form, so the `<` tie-break keeps the same first
   * equal-cost edge and the decoded bits are identical. */
  for (int state = 0; state < num_states; ++state) {
    const size_t source_row = (size_t)state * drift_width;
    for (int drift_index = 0; drift_index < drift_width; ++drift_index) {
      const double current_cost = tr->metric[source_row + drift_index];
      if (current_cost == INFINITY) {
        continue;
      }
      /* Ending drift next_drift_index consumes n + (next_drift_index -
       * drift_index) bits; that must stay in [0, max_consume]. The high end is
       * always satisfied (next_drift_index <= drift_width-1 = 2*max_drift), so
       * only the low bound clamps the loop start. */
      int first = drift_index - n;
      if (first < 0) {
        first = 0;
      }
      for (int bit = 0; bit <= 1; ++bit) {
        const int edge = state * 2 + bit;
        const int next_state = code->next_state[edge];
        const double *final_row =
            tr->align_final +
            ((size_t)tr->group_of[edge] * drift_width + drift_index) * stride;
        const size_t dest_row = (size_t)next_state * drift_width;
        for (int next_drift_index = first; next_drift_index < drift_width;
             ++next_drift_index) {
          const double branch_cost = final_row[n + (next_drift_index - drift_index)];
          if (branch_cost == INFINITY) {
            continue;
          }
          const double cost = current_cost + branch_cost;
          const size_t destination = dest_row + next_drift_index;
          if (cost < tr->next_metric[destination]) {
            tr->next_metric[destination] = cost;
            layer[destination] =
                dv_bp_pack(state, drift_index, (unsigned int)bit);
          }
        }
      }
    }
  }

  const double increment = normalize(tr->next_metric, count);
  const double alpha = 2.0 / (ctx->decision_depth + 1.0);
  tr->smoothed_cost += alpha * (increment - tr->smoothed_cost);

  double *temp = tr->metric;
  tr->metric = tr->next_metric;
  tr->next_metric = temp;
}

/* Fast path for max_drift == 0 with indels disabled (cost_ins == cost_del ==
 * +inf, i.e. p_ins == p_del == 0). The drift window is one wide and the
 * alignment is forced diagonal, so an edge's cost is just the sum of its n
 * per-bit match costs - a plain Viterbi butterfly with no alignment DP and no
 * drift loops. This yields exactly the general path's branch cost (final_row[n])
 * in the same (state, bit) order, so output is identical. */
static void forward_pass_nodrift(dv_decode_ctx *ctx, dv_trellis *tr) {
  const dv_code *code = tr->code;
  const int n = ctx->n, num_states = ctx->num_states;
  const int base = ctx->read_base;
  const int in_range = base >= 0 && base + n <= ctx->received_length;
  const double keep_total = (double)n * ctx->cost_keep;

  for (int state = 0; state < num_states; ++state) {
    tr->next_metric[state] = INFINITY;
  }
  dv_backpointer *layer =
      tr->backpointers +
      (size_t)(ctx->steps % ctx->decision_depth) * (size_t)num_states;

  for (int state = 0; state < num_states; ++state) {
    const double current_cost = tr->metric[state];
    if (current_cost == INFINITY) {
      continue;
    }
    for (int bit = 0; bit <= 1; ++bit) {
      const int edge = state * 2 + bit;
      double branch_cost = INFINITY;
      if (in_range) {
        const uint8_t *expected = &code->output[(size_t)edge * n];
        branch_cost = keep_total;
        for (int j = 0; j < n; ++j) {
          const uint8_t received_bit = ctx->received[base + j];
          branch_cost += received_bit == DV_ERASURE       ? ctx->cost_erase
                         : received_bit == expected[j]     ? ctx->cost_match
                                                           : ctx->cost_miss;
        }
      }
      if (branch_cost == INFINITY) {
        continue;
      }
      const double cost = current_cost + branch_cost;
      const int next_state = code->next_state[edge];
      if (cost < tr->next_metric[next_state]) {
        tr->next_metric[next_state] = cost;
        layer[next_state] = dv_bp_pack(state, 0, (unsigned int)bit);
      }
    }
  }

  const double increment = normalize(tr->next_metric, (size_t)num_states);
  const double alpha = 2.0 / (ctx->decision_depth + 1.0);
  tr->smoothed_cost += alpha * (increment - tr->smoothed_cost);

  double *temp = tr->metric;
  tr->metric = tr->next_metric;
  tr->next_metric = temp;
}

/* Advance every trellis one fused step. The re-anchor sigma is decided once -
 * from the best-fitting trellis (lowest smoothed cost), so a non-matching code
 * never steers the shared timing - and applied to every trellis's window plus
 * the one shared read cursor, after which each runs its own forward pass. With
 * a single trellis this is the ordinary per-code step. */
void dv_decode_step(dv_decode_ctx *ctx, dv_trellis *trs, size_t n) {
  size_t lead = 0;
  for (size_t j = 1; j < n; ++j) {
    if (trs[j].smoothed_cost < trs[lead].smoothed_cost) {
      lead = j;
    }
  }

  const int sigma = pick_shift(ctx, &trs[lead]);
  if (sigma != 0) {
    for (size_t j = 0; j < n; ++j) {
      reanchor_metric(ctx, &trs[j], sigma);
    }
    ctx->read_base += sigma;
  }
  ctx->shift[ctx->steps % ctx->decision_depth] = sigma;

  /* With no drift window and indels disabled the alignment is forced diagonal;
   * a dedicated butterfly skips the DP entirely (identical result). */
  const int nodrift = ctx->max_drift == 0 && ctx->cost_ins == INFINITY &&
                      ctx->cost_del == INFINITY;
  for (size_t j = 0; j < n; ++j) {
    if (nodrift) {
      forward_pass_nodrift(ctx, &trs[j]);
    } else {
      forward_pass(ctx, &trs[j]);
    }
  }

  ctx->steps++;
  ctx->read_base += ctx->n;
  if (ctx->read_base - ctx->max_drift >= ctx->received_capacity / 2) {
    compact_received(ctx);
  }
}

/* Walk the backpointers from frontier node `frontier` back to step `target` and
 * return the input bit decided there. Each step's backpointer is stored in that
 * step's drift frame, so when stepping back across a re-anchor we shift the
 * predecessor's drift index by that step's recorded sigma. */
unsigned char dv_trellis_trace(const dv_decode_ctx *ctx, const dv_trellis *tr,
                               int frontier, long long target) {
  const size_t count = (size_t)ctx->num_states * ctx->drift_width;
  /* Traceback only reaches back through retained ring layers; a deeper target
   * would read a backpointer/shift slot the next step has already overwritten. */
  DV_ASSERT(ctx->steps - target <= ctx->decision_depth);
  int node = frontier;
  unsigned char bit = 0;
  for (long long i = ctx->steps - 1; i >= target; --i) {
    const dv_backpointer *layer =
        tr->backpointers + (size_t)(i % ctx->decision_depth) * count;
    const dv_backpointer entry = layer[node];
    if (i == target) {
      bit = (unsigned char)dv_bp_bit(entry);
      break;
    }
    const int prev_drift_index =
        dv_bp_drift(entry) + ctx->shift[i % ctx->decision_depth];
    /* In-bounds by construction: reanchor_metric empties the slot that would
     * translate out of the window, so it is never a recorded predecessor. */
    DV_ASSERT(prev_drift_index >= 0 && prev_drift_index < ctx->drift_width);
    node = dv_bp_state(entry) * ctx->drift_width + prev_drift_index;
  }
  return bit;
}

/* Probability that the decoder is locked onto a valid stream of THIS specific
 * code, from the best surviving path's cost rate (a smoothed average, see
 * forward_pass). A path can only stay cheap if the received bits actually fit
 * this code's codewords; a different code's stream - even one of the same rate
 * and constraint length - forces mismatches that raise the cost. Map the
 * smoothed cost linearly from expected_lock (a clean lock -> ~1) to
 * expected_unlock (enough misfit that it clearly is not this code -> ~0).
 *
 * This is what makes the value code-specific rather than a generic "is some
 * path dominant?" confidence: a confidently decoded WRONG code is dominant but
 * expensive, so it reads as unlocked. (Two encoders for the SAME code are not
 * "wrong" - they share codewords - and correctly read as locked.) */
double dv_trellis_lock(const dv_decode_ctx *ctx, const dv_trellis *tr) {
  const double gap = ctx->expected_unlock - ctx->expected_lock;
  if (gap <= 0.0) {
    return 0.0;
  }
  double probability = (ctx->expected_unlock - tr->smoothed_cost) / gap;
  if (probability < 0.0) {
    probability = 0.0;
  } else if (probability > 1.0) {
    probability = 1.0;
  }
  return probability;
}

/* Process steps for one trellis, emitting forced decisions, until input/output
 * limits hit. Each emitted bit's lock probability is written to lock_out[] when
 * non-NULL. `draining` relaxes the look-ahead requirement for end-of-stream. */
static int run(dv_decode_ctx *ctx, dv_trellis *tr, uint8_t *out,
               double *lock_out, int max_out, int draining) {
  int output_count = 0;
  for (;;) {
    if (!draining) {
      /* +1 of slack covers a re-anchor stepping the cursor forward. */
      if (ctx->received_length < ctx->read_base + ctx->n + ctx->max_drift + 1) {
        break; /* not enough look-ahead yet */
      }
    } else {
      if (ctx->received_length - ctx->read_base < ctx->n) {
        break; /* less than one group left  */
      }
    }

    /* Processing the next step overwrites the backpointer layer of step
     * (steps - decision_depth), so its decision must be emitted first. */
    if (ctx->steps >= ctx->decision_depth) {
      if (output_count >= max_out) {
        break;
      }
      if (lock_out) {
        lock_out[output_count] = dv_trellis_lock(ctx, tr);
      }
      out[output_count++] =
          dv_trellis_trace(ctx, tr, dv_trellis_frontier(ctx, tr), ctx->decided);
      ctx->decided++;
    }
    dv_decode_step(ctx, tr, 1);
  }
  return output_count;
}

/* Initialise a trellis's metric for blind acquisition: every encoder state is
 * equally likely (all at zero drift), so the decoder locks on whether it starts
 * at the stream's beginning or is tapped partway through. */
static void init_metric(const dv_decode_ctx *ctx, dv_trellis *tr) {
  const size_t count = (size_t)ctx->num_states * ctx->drift_width;
  for (size_t i = 0; i < count; ++i) {
    tr->metric[i] = INFINITY;
  }
  for (int state = 0; state < ctx->num_states; ++state) {
    tr->metric[node_at(state, ctx->max_drift, ctx->drift_width)] =
        0.0; /* zero drift, cost 0 */
  }
}

/* -- internal context / trellis lifecycle ---------------------------------- */

int dv_decode_ctx_init(dv_decode_ctx *ctx, const dv_stream_params *params,
                       const dv_code *code) {
  if (!ctx || !params || !code) {
    return DV_ERR_ARG;
  }
  const int decision_depth = params->decision_depth;
  const int max_drift = params->max_drift;
  const double p_sub = params->p_sub;
  const double p_ins = params->p_ins;
  const double p_del = params->p_del;
  const double p_erase = params->p_erase;

  if (decision_depth < 1 || max_drift < 0) {
    return DV_ERR_ARG;
  }
  if (!(p_sub > 0.0 && p_sub < 1.0) || !(p_erase >= 0.0 && p_erase < 1.0) ||
      !(p_ins + p_del < 1.0) || p_ins < 0.0 || p_del < 0.0) {
    return DV_ERR_ARG;
  }
  /* Insertion/deletion probabilities are only consulted when tracking drift;
   * with max_drift == 0 they may be left 0 (correct flips only). */
  if (max_drift > 0 && (p_ins <= 0.0 || p_del <= 0.0)) {
    return DV_ERR_ARG;
  }

  ctx->n = code->n;
  ctx->max_drift = max_drift;
  ctx->num_states = code->n_states;
  ctx->drift_width = 2 * max_drift + 1;
  ctx->decision_depth = decision_depth;

  /* Channel model: a coded bit is erased with prob p_erase; otherwise it is
   * received and flipped with prob p_sub. The common (1 - p_erase) factor is
   * kept explicit so paths reading different erasure counts compare correctly
   * (p_erase = 0 reduces these to the plain hard-decision metric). */
  ctx->cost_match = -dv_log((1.0 - p_erase) * (1.0 - p_sub));
  ctx->cost_miss = -dv_log((1.0 - p_erase) * p_sub);
  ctx->cost_erase = -dv_log(p_erase); /* +inf when p_erase == 0 (never read) */
  ctx->cost_keep = -dv_log(1.0 - p_ins - p_del);
  ctx->cost_ins = -dv_log(p_ins);
  ctx->cost_del = -dv_log(p_del);

  /* Lock anchors (per step = n coded bits). A "kept" coded bit costs cost_keep
   * plus a match/miss term; the expected misfit fraction is p_sub when locked,
   * and we call it "unlocked" once misfit reaches the midpoint between p_sub
   * and 0.5 (random). Erased bits contribute cost_erase to both. */
  const double misfit_lock = p_sub;
  const double misfit_unlock = 0.5 * (p_sub + 0.5);
  const double erase_term = p_erase > 0.0 ? p_erase * ctx->cost_erase : 0.0;
  const double kept = 1.0 - p_erase;
  ctx->expected_lock = ctx->n * (ctx->cost_keep + erase_term +
                                 kept * ((1.0 - misfit_lock) * ctx->cost_match +
                                         misfit_lock * ctx->cost_miss));
  ctx->expected_unlock =
      ctx->n * (ctx->cost_keep + erase_term +
                kept * ((1.0 - misfit_unlock) * ctx->cost_match +
                        misfit_unlock * ctx->cost_miss));

  ctx->steps = 0;
  ctx->decided = 0;
  ctx->read_base = 0;
  ctx->received_length = 0;
  ctx->received_capacity =
      (ctx->decision_depth + 2) * ctx->n + 8 * ctx->max_drift + 64;

  ctx->shift = dv_malloc((size_t)ctx->decision_depth * sizeof(int));
  ctx->received = dv_malloc((size_t)ctx->received_capacity);
  if (!ctx->shift || !ctx->received) {
    dv_decode_ctx_free(ctx);
    return DV_ERR_ALLOC;
  }
  return DV_OK;
}

void dv_decode_ctx_free(dv_decode_ctx *ctx) {
  if (!ctx) {
    return;
  }
  dv_free(ctx->shift);
  dv_free(ctx->received);
  ctx->shift = NULL;
  ctx->received = NULL;
}

int dv_trellis_init(dv_trellis *tr, const dv_decode_ctx *ctx,
                    const dv_code *code) {
  if (!tr || !ctx || !code) {
    return DV_ERR_ARG;
  }
  /* The packed backpointer (decode_internal.h) carries prev_state in 16 bits and
   * prev_drift_index in 15. dv_code_create caps K (num_states <= 256) and real
   * max_drift is tiny, but guard the packing invariant rather than silently
   * truncating. */
  if (ctx->num_states - 1 > DV_BP_MAX_STATE ||
      ctx->drift_width - 1 > DV_BP_MAX_DRIFT) {
    return DV_ERR_ARG;
  }
  tr->code = code;
  tr->smoothed_cost =
      ctx->expected_unlock; /* assume unlocked until the stream proves it */

  const size_t count = (size_t)ctx->num_states * ctx->drift_width;
  const int max_consume = ctx->n + 2 * ctx->max_drift;
  const int window = ctx->n + 4 * ctx->max_drift; /* match-cost table width */
  const int edges = ctx->num_states * 2;
  tr->metric = dv_malloc(count * sizeof(double));
  tr->next_metric = dv_malloc(count * sizeof(double));
  tr->backpointers =
      dv_malloc((size_t)ctx->decision_depth * count * sizeof(dv_backpointer));
  tr->alignment =
      dv_malloc((size_t)(ctx->n + 1) * (max_consume + 1) * sizeof(double));
  tr->group_of = dv_malloc((size_t)edges * sizeof(int));
  tr->group_expected = dv_malloc((size_t)edges * sizeof(const uint8_t *));
  tr->match_cost0 = dv_malloc((size_t)window * sizeof(double));
  tr->match_cost1 = dv_malloc((size_t)window * sizeof(double));
  tr->in_range = dv_malloc((size_t)window * sizeof(signed char));
  if (!tr->metric || !tr->next_metric || !tr->backpointers || !tr->alignment ||
      !tr->group_of || !tr->group_expected || !tr->match_cost0 ||
      !tr->match_cost1 || !tr->in_range) {
    dv_trellis_free(tr);
    return DV_ERR_ALLOC;
  }

  /* Map each (state, bit) edge to its distinct n-bit output group, keeping one
   * representative output row per group for the per-step alignment precompute. */
  tr->n_groups = 0;
  for (int edge = 0; edge < edges; ++edge) {
    const uint8_t *row = &code->output[(size_t)edge * ctx->n];
    int found = -1;
    for (int g = 0; g < tr->n_groups; ++g) {
      int same = 1;
      for (int j = 0; j < ctx->n; ++j) {
        if (tr->group_expected[g][j] != row[j]) {
          same = 0;
          break;
        }
      }
      if (same) {
        found = g;
        break;
      }
    }
    if (found < 0) {
      found = tr->n_groups;
      tr->group_expected[tr->n_groups++] = row;
    }
    tr->group_of[edge] = found;
  }

  tr->align_final = dv_malloc((size_t)tr->n_groups * ctx->drift_width *
                              (max_consume + 1) * sizeof(double));
  if (!tr->align_final) {
    dv_trellis_free(tr);
    return DV_ERR_ALLOC;
  }

  init_metric(ctx, tr);
  return DV_OK;
}

void dv_trellis_free(dv_trellis *tr) {
  if (!tr) {
    return;
  }
  dv_free(tr->metric);
  dv_free(tr->next_metric);
  dv_free(tr->backpointers);
  dv_free(tr->alignment);
  dv_free(tr->group_of);
  dv_free(tr->group_expected);
  dv_free(tr->align_final);
  dv_free(tr->match_cost0);
  dv_free(tr->match_cost1);
  dv_free(tr->in_range);
  tr->metric = NULL;
  tr->next_metric = NULL;
  tr->backpointers = NULL;
  tr->alignment = NULL;
  tr->group_of = NULL;
  tr->group_expected = NULL;
  tr->align_final = NULL;
  tr->match_cost0 = NULL;
  tr->match_cost1 = NULL;
  tr->in_range = NULL;
}

int dv_decode_feed(dv_decode_ctx *ctx, const uint8_t *in, int n_in) {
  int status = reserve_received(ctx, n_in);
  if (status < 0) {
    return status;
  }
  dv_memcpy(ctx->received + ctx->received_length, in, (size_t)n_in);
  ctx->received_length += n_in;
  return DV_OK;
}

/* -- public single-stream decoder ------------------------------------------ */

/* The public decoder is one shared context plus one trellis: the N == 1 case of
 * the fused machinery above. */
struct dv_stream_decoder {
  dv_decode_ctx ctx;
  dv_trellis trellis;
};

dv_stream_decoder *dv_stream_decoder_create(const dv_code *code,
                                            const dv_stream_params *params) {
  if (!code || !params) {
    return NULL;
  }
  struct dv_stream_decoder *sd = dv_calloc(1, sizeof(*sd));
  if (!sd) {
    return NULL;
  }
  if (dv_decode_ctx_init(&sd->ctx, params, code) < 0) {
    dv_free(sd);
    return NULL;
  }
  if (dv_trellis_init(&sd->trellis, &sd->ctx, code) < 0) {
    dv_decode_ctx_free(&sd->ctx);
    dv_free(sd);
    return NULL;
  }
  return sd;
}

void dv_stream_decoder_destroy(dv_stream_decoder *sd) {
  if (!sd) {
    return;
  }
  dv_trellis_free(&sd->trellis);
  dv_decode_ctx_free(&sd->ctx);
  dv_free(sd);
}

int dv_stream_decode(dv_stream_decoder *sd, const uint8_t *in, int n_in,
                     uint8_t *out, double *lock_probability, int max_out) {
  if (!sd || (n_in > 0 && !in) || n_in < 0 || (max_out > 0 && !out) ||
      max_out < 0) {
    return DV_ERR_ARG;
  }
  int status = dv_decode_feed(&sd->ctx, in, n_in);
  if (status < 0) {
    return status;
  }
  return run(&sd->ctx, &sd->trellis, out, lock_probability, max_out,
             /*draining=*/0);
}

int dv_stream_decode_flush(dv_stream_decoder *sd, uint8_t *out, int max_out) {
  if (!sd || (max_out > 0 && !out) || max_out < 0) {
    return DV_ERR_ARG;
  }
  int output_count =
      run(&sd->ctx, &sd->trellis, out, /*lock=*/NULL, max_out, /*draining=*/1);

  /* Drain the pipeline: decide the remaining buffered steps from the final
   * frontier (reduced traceback depth for the last <= decision_depth bits). */
  if (sd->ctx.decided < sd->ctx.steps && output_count < max_out) {
    int frontier = dv_trellis_frontier(&sd->ctx, &sd->trellis);
    while (sd->ctx.decided < sd->ctx.steps && output_count < max_out) {
      out[output_count++] =
          dv_trellis_trace(&sd->ctx, &sd->trellis, frontier, sd->ctx.decided);
      sd->ctx.decided++;
    }
  }
  return output_count;
}
