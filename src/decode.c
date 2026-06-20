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

#include <math.h>
#include <stdlib.h>
#include <string.h>

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
 */
/* clang-format on */

/* Backpointer for one node: where it came from (prev_state, prev_drift_index)
 * and the input bit that got there. */
typedef struct {
  int prev_state;
  int prev_drift_index;
  unsigned char bit;
} dv_backpointer;

/* clang-format off */
struct dv_stream_decoder {
  const dv_code *code; /* borrowed                                          */
  int n, max_drift, num_states, drift_width, decision_depth;
  /* Branch-metric constants, in cost (negative-log-likelihood) units. */
  double cost_match, cost_miss, cost_erase, cost_keep, cost_ins, cost_del;
  /* Lock detection: per-step best-path cost expected when the input really is
   * this code's stream (expected_lock) vs. when a quarter-ish of coded bits
   * don't fit it (expected_unlock), and a smoothed (EWMA) observed cost. */
  double expected_lock, expected_unlock, smoothed_cost;

  double *metric;               /* [num_states*drift_width] node costs      */
  double *next_metric;          /* [num_states*drift_width] scratch         */
  dv_backpointer *backpointers; /* [decision_depth*num_states*drift_width]  */
  int *shift;                   /* [decision_depth] re-anchor sigma/step    */
  double *alignment;            /* [(n+1)*(n+2*max_drift+1)] DP scratch     */

  long long steps;   /* trellis steps processed                      */
  long long decided; /* decisions emitted (next step index to emit)  */

  /* Received-bit buffer: valid bits live in received[0 .. received_length).
   * read_base is the buffer index of the current step's zero-drift read base. */
  uint8_t *received;
  int received_capacity, received_length, read_base;
};
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
static void compact_received(struct dv_stream_decoder *sd) {
  int keep_from = sd->read_base - 2 * sd->max_drift;
  if (keep_from <= 0) {
    return;
  }
  memmove(sd->received, sd->received + keep_from,
          (size_t)(sd->received_length - keep_from));
  sd->received_length -= keep_from;
  sd->read_base -= keep_from;
}

/* Ensure room for `extra` more bits, compacting then growing as needed. */
static int reserve_received(struct dv_stream_decoder *sd, int extra) {
  compact_received(sd);
  if (sd->received_length + extra > sd->received_capacity) {
    int new_capacity = sd->received_capacity * 2;
    if (new_capacity < sd->received_length + extra) {
      new_capacity = sd->received_length + extra;
    }
    uint8_t *new_buffer = realloc(sd->received, (size_t)new_capacity);
    if (!new_buffer) {
      return DV_ERR_ALLOC;
    }
    sd->received = new_buffer;
    sd->received_capacity = new_capacity;
  }
  return DV_OK;
}

/* -- core trellis ---------------------------------------------------------- */

/* Flat index of the lowest-cost node at the current frontier (the node states
 * one step past the last one processed). */
static int frontier_best(const struct dv_stream_decoder *sd) {
  const size_t count = (size_t)sd->num_states * sd->drift_width;
  double best_cost = INFINITY;
  int best = 0;
  for (size_t i = 0; i < count; ++i) {
    if (sd->metric[i] < best_cost) {
      best_cost = sd->metric[i];
      best = (int)i;
    }
  }
  return best;
}

/* Choose this step's re-anchor shift: nudge the window one step toward centre
 * when the best node's drift leaves a deadband, so its drift stays well inside
 * the window even as cumulative drift grows. */
static int pick_shift(const struct dv_stream_decoder *sd) {
  if (sd->max_drift == 0) {
    return 0; /* no drift tracking: window is one wide, nothing to shift */
  }
  const int best_drift_index = frontier_best(sd) % sd->drift_width;
  const int drift = best_drift_index - sd->max_drift;
  const int deadband = (sd->max_drift + 1) / 2;
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
static void align_fill(const struct dv_stream_decoder *sd,
                       const uint8_t *expected, int base) {
  const int n = sd->n, max_consume = sd->n + 2 * sd->max_drift,
            stride = max_consume + 1;
  double *cost_table = sd->alignment;

  cost_table[0] = 0.0;
  for (int consumed = 1; consumed <= max_consume; ++consumed) {
    const int buffer_index = base + consumed - 1;
    cost_table[consumed] =
        (buffer_index >= 0 && buffer_index < sd->received_length)
            ? cost_table[consumed - 1] + sd->cost_ins
            : INFINITY;
  }
  for (int j = 1; j <= n; ++j) {
    double *cost_row = cost_table + (size_t)j * stride;
    const double *prev_row = cost_table + (size_t)(j - 1) * stride;
    cost_row[0] =
        prev_row[0] + sd->cost_del; /* delete expected[j-1], consume nothing */
    for (int consumed = 1; consumed <= max_consume; ++consumed) {
      double best =
          prev_row[consumed] + sd->cost_del; /* deletion always avail. */
      const int buffer_index = base + consumed - 1;
      if (buffer_index >= 0 && buffer_index < sd->received_length) {
        const uint8_t received_bit = sd->received[buffer_index];
        const double match_cost =
            sd->cost_keep + (received_bit == DV_ERASURE        ? sd->cost_erase
                             : received_bit == expected[j - 1] ? sd->cost_match
                                                               : sd->cost_miss);
        const double align_cost =
            prev_row[consumed - 1] + match_cost; /* match / substitute */
        const double insert_cost =
            cost_row[consumed - 1] + sd->cost_ins; /* extra received bit */
        if (align_cost < best) best = align_cost;
        if (insert_cost < best) best = insert_cost;
      }
      cost_row[consumed] = best;
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

/* Shift the drift window by `sigma` (each node's drift index drift_index ->
 * drift_index - sigma) and advance the read cursor to match, so the bits
 * actually read don't change but the live drift values re-centre on the current
 * timing. Nodes shifted out of the window are dropped. Uses next_metric as
 * scratch. */
static void reanchor(struct dv_stream_decoder *sd, int sigma) {
  const int num_states = sd->num_states, drift_width = sd->drift_width;
  for (int state = 0; state < num_states; ++state) {
    for (int drift_index = 0; drift_index < drift_width; ++drift_index) {
      const int source = drift_index + sigma;
      sd->next_metric[node_at(state, drift_index, drift_width)] =
          (source >= 0 && source < drift_width)
              ? sd->metric[node_at(state, source, drift_width)]
              : INFINITY;
    }
  }
  memcpy(sd->metric, sd->next_metric,
         (size_t)num_states * drift_width * sizeof(double));
  sd->read_base += sigma;
}

/* Advance one trellis step: re-anchor if needed, run the forward pass from
 * sd->metric into sd->next_metric (recording backpointers), normalise, and
 * swap. Branches that would read outside the buffered window are skipped, which
 * is what makes the ends of the stream safe. */
static void stream_step(struct dv_stream_decoder *sd) {
  const dv_code *code = sd->code;
  const int n = sd->n, max_drift = sd->max_drift, num_states = sd->num_states,
            drift_width = sd->drift_width;
  const size_t count = (size_t)num_states * drift_width;
  const int max_consume =
      sd->n + 2 * sd->max_drift; /* most received bits a branch can consume */

  const int sigma = pick_shift(sd);
  if (sigma != 0) {
    reanchor(sd, sigma);
  }
  sd->shift[sd->steps % sd->decision_depth] = sigma;

  for (size_t i = 0; i < count; ++i) {
    sd->next_metric[i] = INFINITY;
  }
  dv_backpointer *layer =
      sd->backpointers + (size_t)(sd->steps % sd->decision_depth) * count;

  /* For every live node, try both input bits; the bit-level alignment DP then
   * spreads cost to every reachable ending drift in one pass. */
  for (int state = 0; state < num_states; ++state) {
    for (int drift_index = 0; drift_index < drift_width; ++drift_index) {
      const double current_cost =
          sd->metric[node_at(state, drift_index, drift_width)];
      if (current_cost == INFINITY) {
        continue;
      }
      const int base = sd->read_base + (drift_index - max_drift);

      for (int bit = 0; bit <= 1; ++bit) {
        const int next_state = code->next_state[state * 2 + bit];
        const uint8_t *expected =
            &code->output[((size_t)(state * 2 + bit)) * n];
        align_fill(sd, expected, base);
        const double *final_row = sd->alignment + (size_t)n * (max_consume + 1);

        /* Ending drift next_drift_index consumes n + (next_drift_index -
         * drift_index) received bits; final_row holds that consumption's cost.
         */
        for (int next_drift_index = 0; next_drift_index < drift_width;
             ++next_drift_index) {
          const int consumed = n + (next_drift_index - drift_index);
          if (consumed < 0 || consumed > max_consume) {
            continue;
          }
          const double branch_cost = final_row[consumed];
          if (branch_cost == INFINITY) {
            continue;
          }
          const double cost = current_cost + branch_cost;
          const size_t destination =
              node_at(next_state, next_drift_index, drift_width);
          if (cost < sd->next_metric[destination]) {
            sd->next_metric[destination] = cost;
            layer[destination].prev_state = state;
            layer[destination].prev_drift_index = drift_index;
            layer[destination].bit = (unsigned char)bit;
          }
        }
      }
    }
  }

  const double increment = normalize(sd->next_metric, count);
  const double alpha = 2.0 / (sd->decision_depth + 1.0);
  sd->smoothed_cost += alpha * (increment - sd->smoothed_cost);

  double *temp = sd->metric;
  sd->metric = sd->next_metric;
  sd->next_metric = temp;
  sd->steps++;
  sd->read_base += n;
}

/* Walk the backpointers from frontier node `frontier` back to step `target` and
 * return the input bit decided there. Each step's backpointer is stored in that
 * step's drift frame, so when stepping back across a re-anchor we shift the
 * predecessor's drift index by that step's recorded sigma. */
static unsigned char trace_bit(const struct dv_stream_decoder *sd, int frontier,
                               long long target) {
  const size_t count = (size_t)sd->num_states * sd->drift_width;
  int node = frontier;
  unsigned char bit = 0;
  for (long long i = sd->steps - 1; i >= target; --i) {
    const dv_backpointer *layer =
        sd->backpointers + (size_t)(i % sd->decision_depth) * count;
    const dv_backpointer entry = layer[node];
    if (i == target) {
      bit = entry.bit;
      break;
    }
    node = entry.prev_state * sd->drift_width +
           (entry.prev_drift_index + sd->shift[i % sd->decision_depth]);
  }
  return bit;
}

/* Probability that the decoder is locked onto a valid stream of THIS specific
 * code, from the best surviving path's cost rate (a smoothed average, see
 * stream_step). A path can only stay cheap if the received bits actually fit
 * this code's codewords; a different code's stream - even one of the same rate
 * and constraint length - forces mismatches that raise the cost. Map the
 * smoothed cost linearly from expected_lock (a clean lock -> ~1) to
 * expected_unlock (enough misfit that it clearly is not this code -> ~0).
 *
 * This is what makes the value code-specific rather than a generic "is some
 * path dominant?" confidence: a confidently decoded WRONG code is dominant but
 * expensive, so it reads as unlocked. (Two encoders for the SAME code are not
 * "wrong" - they share codewords - and correctly read as locked.) */
static double compute_lock_probability(const struct dv_stream_decoder *sd) {
  const double gap = sd->expected_unlock - sd->expected_lock;
  if (gap <= 0.0) {
    return 0.0;
  }
  double probability = (sd->expected_unlock - sd->smoothed_cost) / gap;
  if (probability < 0.0) {
    probability = 0.0;
  } else if (probability > 1.0) {
    probability = 1.0;
  }
  return probability;
}

/* Process steps, emitting forced decisions, until input/output limits hit.
 * Each emitted bit's lock probability is written to lock_out[] when non-NULL.
 * `draining` relaxes the look-ahead requirement for end-of-stream. */
static int run(struct dv_stream_decoder *sd, uint8_t *out, double *lock_out,
               int max_out, int draining) {
  int output_count = 0;
  for (;;) {
    if (!draining) {
      /* +1 of slack covers a re-anchor stepping the cursor forward. */
      if (sd->received_length < sd->read_base + sd->n + sd->max_drift + 1) {
        break; /* not enough look-ahead yet */
      }
    } else {
      if (sd->received_length - sd->read_base < sd->n) {
        break; /* less than one group left  */
      }
    }

    /* Processing the next step overwrites the backpointer layer of step
     * (steps - decision_depth), so its decision must be emitted first. */
    if (sd->steps >= sd->decision_depth) {
      if (output_count >= max_out) {
        break;
      }
      if (lock_out) {
        lock_out[output_count] = compute_lock_probability(sd);
      }
      out[output_count++] = trace_bit(sd, frontier_best(sd), sd->decided);
      sd->decided++;
    }
    stream_step(sd);
    if (sd->read_base - sd->max_drift >= sd->received_capacity / 2) {
      compact_received(sd);
    }
  }
  return output_count;
}

/* Initialise the trellis metrics for blind acquisition: every encoder state is
 * equally likely (all at zero drift), so the decoder locks on whether it starts
 * at the stream's beginning or is tapped partway through. */
static void init_metric(struct dv_stream_decoder *sd) {
  const size_t count = (size_t)sd->num_states * sd->drift_width;
  for (size_t i = 0; i < count; ++i) {
    sd->metric[i] = INFINITY;
  }
  for (int state = 0; state < sd->num_states; ++state) {
    sd->metric[node_at(state, sd->max_drift, sd->drift_width)] =
        0.0; /* zero drift, cost 0 */
  }
}

/* -- public API ------------------------------------------------------------ */

dv_stream_decoder *dv_stream_decoder_create(const dv_code *code,
                                            const dv_stream_params *params) {
  if (!code || !params) {
    return NULL;
  }
  const int decision_depth = params->decision_depth;
  const int max_drift = params->max_drift;
  const double p_sub = params->p_sub;
  const double p_ins = params->p_ins;
  const double p_del = params->p_del;
  const double p_erase = params->p_erase;

  if (decision_depth < 1 || max_drift < 0) {
    return NULL;
  }
  if (!(p_sub > 0.0 && p_sub < 1.0) || !(p_erase >= 0.0 && p_erase < 1.0) ||
      !(p_ins + p_del < 1.0) || p_ins < 0.0 || p_del < 0.0) {
    return NULL;
  }
  /* Insertion/deletion probabilities are only consulted when tracking drift;
   * with max_drift == 0 they may be left 0 (correct flips only). */
  if (max_drift > 0 && (p_ins <= 0.0 || p_del <= 0.0)) {
    return NULL;
  }

  struct dv_stream_decoder *sd = calloc(1, sizeof(*sd));
  if (!sd) {
    return NULL;
  }
  sd->code = code;
  sd->n = code->n;
  sd->max_drift = max_drift;
  sd->num_states = code->n_states;
  sd->drift_width = 2 * max_drift + 1;
  sd->decision_depth = decision_depth;

  /* Channel model: a coded bit is erased with prob p_erase; otherwise it is
   * received and flipped with prob p_sub. The common (1 - p_erase) factor is
   * kept explicit so paths reading different erasure counts compare correctly
   * (p_erase = 0 reduces these to the plain hard-decision metric). */
  sd->cost_match = -log((1.0 - p_erase) * (1.0 - p_sub));
  sd->cost_miss = -log((1.0 - p_erase) * p_sub);
  sd->cost_erase = -log(p_erase); /* +inf when p_erase == 0 (never read) */
  sd->cost_keep = -log(1.0 - p_ins - p_del);
  sd->cost_ins = -log(p_ins);
  sd->cost_del = -log(p_del);

  /* Lock anchors (per step = n coded bits). A "kept" coded bit costs cost_keep
   * plus a match/miss term; the expected misfit fraction is p_sub when locked,
   * and we call it "unlocked" once misfit reaches the midpoint between p_sub
   * and 0.5 (random). Erased bits contribute cost_erase to both. */
  const double misfit_lock = p_sub;
  const double misfit_unlock = 0.5 * (p_sub + 0.5);
  const double erase_term = p_erase > 0.0 ? p_erase * sd->cost_erase : 0.0;
  const double kept = 1.0 - p_erase;
  sd->expected_lock = sd->n * (sd->cost_keep + erase_term +
                               kept * ((1.0 - misfit_lock) * sd->cost_match +
                                       misfit_lock * sd->cost_miss));
  sd->expected_unlock =
      sd->n * (sd->cost_keep + erase_term +
               kept * ((1.0 - misfit_unlock) * sd->cost_match +
                       misfit_unlock * sd->cost_miss));
  sd->smoothed_cost =
      sd->expected_unlock; /* assume unlocked until the stream proves it */

  const size_t count = (size_t)sd->num_states * sd->drift_width;
  sd->metric = malloc(count * sizeof(double));
  sd->next_metric = malloc(count * sizeof(double));
  sd->backpointers =
      malloc((size_t)sd->decision_depth * count * sizeof(dv_backpointer));
  sd->shift = malloc((size_t)sd->decision_depth * sizeof(int));
  sd->alignment = malloc((size_t)(sd->n + 1) * (sd->n + 2 * sd->max_drift + 1) *
                         sizeof(double));
  sd->received_capacity =
      (sd->decision_depth + 2) * sd->n + 8 * sd->max_drift + 64;
  sd->received = malloc((size_t)sd->received_capacity);
  if (!sd->metric || !sd->next_metric || !sd->backpointers || !sd->shift ||
      !sd->alignment || !sd->received) {
    dv_stream_decoder_destroy(sd);
    return NULL;
  }

  init_metric(sd);
  return sd;
}

void dv_stream_decoder_destroy(dv_stream_decoder *sd) {
  if (!sd) {
    return;
  }
  free(sd->metric);
  free(sd->next_metric);
  free(sd->backpointers);
  free(sd->shift);
  free(sd->alignment);
  free(sd->received);
  free(sd);
}

int dv_stream_decode(dv_stream_decoder *sd, const uint8_t *in, int n_in,
                     uint8_t *out, double *lock_probability, int max_out) {
  if (!sd || (n_in > 0 && !in) || n_in < 0 || (max_out > 0 && !out) ||
      max_out < 0) {
    return DV_ERR_ARG;
  }
  int status = reserve_received(sd, n_in);
  if (status < 0) {
    return status;
  }
  memcpy(sd->received + sd->received_length, in, (size_t)n_in);
  sd->received_length += n_in;
  return run(sd, out, lock_probability, max_out, /*draining=*/0);
}

int dv_stream_decode_flush(dv_stream_decoder *sd, uint8_t *out, int max_out) {
  if (!sd || (max_out > 0 && !out) || max_out < 0) {
    return DV_ERR_ARG;
  }
  int output_count = run(sd, out, /*lock=*/NULL, max_out, /*draining=*/1);

  /* Drain the pipeline: decide the remaining buffered steps from the final
   * frontier (reduced traceback depth for the last <= decision_depth bits). */
  if (sd->decided < sd->steps && output_count < max_out) {
    int frontier = frontier_best(sd);
    while (sd->decided < sd->steps && output_count < max_out) {
      out[output_count++] = trace_bit(sd, frontier, sd->decided);
      sd->decided++;
    }
  }
  return output_count;
}
