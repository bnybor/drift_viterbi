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
 * Multi-decoder: decode one received stream against several candidate codes at
 * once and, per output bit, soft-combine their decisions weighted by how well
 * each code fits the stream. Rather than wrap N independent dv_stream_decoders,
 * it drives the decoder internals directly (decode_internal.h): one shared
 * dv_decode_ctx (the single received buffer + cadence) and one dv_trellis per
 * code, all advanced in lockstep by dv_decode_step with one shared re-anchor.
 * Because the cadence is shared, every trellis decides the same step at the same
 * time, so the per-bit merge needs no output realignment - the codes' bits for a
 * given step are directly commensurable and are combined by likelihood weight
 * (see multi_run).
 */

#include <drift_viterbi/multi_decode.h>

#include <drift_viterbi/encode.h>
#include <drift_viterbi/stdlib.h>

#include "decode_internal.h"

/* Defaults for the selection thresholds when params (or a field) is left 0: at
 * least one code's lock probability must clear this floor for any bit to commit,
 * and the winning bit value must lead the other by this share of the total
 * likelihood weight; otherwise the bit is erased. */
static const double DV_MULTI_LOCK_FLOOR_DEFAULT = 0.6;
static const double DV_MULTI_LOCK_MARGIN_DEFAULT = 0.2;

struct dv_multi_decoder {
  dv_decode_ctx ctx;     /* shared received buffer + cadence            */
  dv_trellis *trellises; /* [n] one per code                            */
  size_t n;
  double lock_floor;  /* min lock probability to commit a bit       */
  double lock_margin; /* min lead over the next-best code           */
  /* Index of the most recently locked code, carried across calls so the
   * end-of-stream flush can decode its tail; -1 until one locks. */
  int locked;
};

dv_multi_decoder *dv_multi_create(const dv_multi_params *params) {
  if (!params || (params->codes_len > 0 && !params->codes)) {
    return NULL;
  }
  dv_multi_decoder *m = dv_calloc(1, sizeof(*m));
  if (!m) {
    return NULL;
  }
  m->n = params->codes_len;
  m->lock_floor = params->lock_floor > 0.0 ? params->lock_floor
                                           : DV_MULTI_LOCK_FLOOR_DEFAULT;
  m->lock_margin = params->lock_margin > 0.0 ? params->lock_margin
                                             : DV_MULTI_LOCK_MARGIN_DEFAULT;
  m->locked = -1;

  if (params->codes_len > 0) {
    /* All codes share the one context's dimensions and channel model; take them
     * from the first (they must share a rate - see multi.h). */
    if (dv_decode_ctx_init(&m->ctx, &params->stream, params->codes[0]) < 0) {
      dv_multi_destroy(m);
      return NULL;
    }
    /* calloc so a build failure partway leaves the rest zeroed, and
     * dv_multi_destroy can free what was built (dv_trellis_free is NULL-safe). */
    m->trellises = dv_calloc(params->codes_len, sizeof(dv_trellis));
    if (!m->trellises) {
      dv_multi_destroy(m);
      return NULL;
    }
    for (size_t j = 0; j < params->codes_len; ++j) {
      /* Every trellis is sized from the shared ctx (taken from codes[0]) yet
       * forward_pass walks each code's own next_state/output tables over
       * ctx->num_states. Codes that differ in rate (dv_code_n) or constraint
       * length (dv_code_k, hence n_states) would index those tables out of
       * bounds, so reject the set rather than corrupt memory. A NULL slot fails
       * here too: dv_code_n(NULL) == -1 != codes[0]'s rate. */
      if (dv_code_n(params->codes[j]) != dv_code_n(params->codes[0]) ||
          dv_code_k(params->codes[j]) != dv_code_k(params->codes[0])) {
        dv_multi_destroy(m);
        return NULL;
      }
      if (dv_trellis_init(&m->trellises[j], &m->ctx, params->codes[j]) < 0) {
        dv_multi_destroy(m);
        return NULL;
      }
    }
  }
  return m;
}

void dv_multi_destroy(dv_multi_decoder *m) {
  if (!m) {
    return;
  }
  if (m->trellises) {
    for (size_t j = 0; j < m->n; ++j) {
      dv_trellis_free(&m->trellises[j]);
    }
    dv_free(m->trellises);
  }
  dv_decode_ctx_free(&m->ctx);
  dv_free(m);
}

/* Common argument check for the two public entry points; mirrors
 * dv_stream_decode's contract. */
static int multi_args_ok(const dv_multi_decoder *d, const uint8_t *in, int n_in,
                         const uint8_t *out, int max_out) {
  return d && !(n_in > 0 && !in) && n_in >= 0 && !(max_out > 0 && !out) &&
         max_out >= 0 && !(d->n > 0 && !d->trellises);
}

/* Step the fused trellises over the buffered input, emitting one merged bit per
 * decided step: the likelihood-weighted soft combination of the codes' traced
 * bits (see the loop body), or DV_ERASURE when nothing is locked or the combined
 * vote is too close to call. `draining` relaxes the look-ahead for end-of-stream.
 * Shared by decode and flush. */
static int multi_run(dv_multi_decoder *d, uint8_t *out, int *locked_decoder,
                     int max_out, int draining) {
  dv_decode_ctx *ctx = &d->ctx;
  int output_count = 0;
  int last = d->locked;
  for (;;) {
    if (!draining) {
      if (ctx->received_length < ctx->read_base + ctx->n + ctx->max_drift + 1) {
        break;
      }
    } else {
      if (ctx->received_length - ctx->read_base < ctx->n) {
        break;
      }
    }

    if (ctx->steps >= ctx->decision_depth) {
      if (output_count >= max_out) {
        break;
      }
      /* Likelihood-weighted soft combine across the codes. Each trellis's
       * smoothed_cost is the best path's per-step negative-log-likelihood under
       * that code, so w_j = exp(-(cost_j - min_cost)) is code j's likelihood
       * relative to the best-fitting one: a code that does not fit (much higher
       * cost) gets vanishing weight automatically, with no hard cutoff. Every
       * code then casts its own traced bit and we accumulate the weight behind
       * each bit value; the heavier value wins when it leads by >= lock_margin
       * of the total weight, else the bit is genuinely ambiguous and erased.
       *
       * One code is still gated absolutely: unless the best-locked code clears
       * lock_floor, nothing is locked (e.g. noise, or between codes) and the bit
       * is erased regardless of how the weights split. The reported winner is
       * the single likeliest code (lowest cost), or -1 when erased. */
      double min_cost = d->trellises[0].smoothed_cost;
      int best_idx = 0;
      for (size_t j = 1; j < d->n; ++j) {
        if (d->trellises[j].smoothed_cost < min_cost) {
          min_cost = d->trellises[j].smoothed_cost;
          best_idx = (int)j;
        }
      }
      const double best_lock = dv_trellis_lock(ctx, &d->trellises[best_idx]);

      int decided_bit = -1; /* 0/1 once chosen; stays -1 to erase */
      if (best_lock >= d->lock_floor) {
        double weight_total = 0.0, weight_ones = 0.0;
        for (size_t j = 0; j < d->n; ++j) {
          const double w = dv_exp(min_cost - d->trellises[j].smoothed_cost);
          weight_total += w;
          const int frontier = dv_trellis_frontier(ctx, &d->trellises[j]);
          if (dv_trellis_trace(ctx, &d->trellises[j], frontier, ctx->decided)) {
            weight_ones += w;
          }
        }
        /* margin between the two bit values, in units of total weight:
         * (W1 - W0) / Wtot = 2*W1/Wtot - 1, range [-1, 1]. */
        const double lead = 2.0 * weight_ones / weight_total - 1.0;
        if (lead >= d->lock_margin) {
          decided_bit = 1;
        } else if (-lead >= d->lock_margin) {
          decided_bit = 0;
        }
      }

      if (decided_bit >= 0) {
        out[output_count] = (uint8_t)decided_bit;
        last = best_idx;
        if (locked_decoder) {
          locked_decoder[output_count] = best_idx;
        }
      } else {
        out[output_count] = DV_ERASURE;
        if (locked_decoder) {
          locked_decoder[output_count] = -1;
        }
      }
      output_count++;
      ctx->decided++;
    }
    dv_decode_step(ctx, d->trellises, d->n);
  }
  d->locked = last; /* carry the latest winner into the end-of-stream flush */
  return output_count;
}

int dv_multi_decode(dv_multi_decoder *d, const uint8_t *in, int n_in,
                    uint8_t *out, int *locked_decoder, int max_out) {
  if (!multi_args_ok(d, in, n_in, out, max_out)) {
    return DV_ERR_ARG;
  }
  if (d->n == 0) {
    return 0;
  }
  int status = dv_decode_feed(&d->ctx, in, n_in);
  if (status < 0) {
    return status;
  }
  return multi_run(d, out, locked_decoder, max_out, /*draining=*/0);
}

int dv_multi_decode_flush(dv_multi_decoder *d, uint8_t *out, int max_out) {
  if (!multi_args_ok(d, NULL, 0, out, max_out)) {
    return DV_ERR_ARG;
  }
  if (d->n == 0 || max_out == 0) {
    return 0;
  }
  dv_decode_ctx *ctx = &d->ctx;
  int output_count = multi_run(d, out, /*locked_decoder=*/NULL, max_out,
                               /*draining=*/1);

  /* Drain the pipeline. The last decision_depth bits carry no lock probability,
   * so decode them with the most recently locked code (its trellis holds the
   * true tail); with nothing ever locked, erase them. */
  if (ctx->decided < ctx->steps && output_count < max_out) {
    int win = d->locked;
    if (win < 0) {
      while (ctx->decided < ctx->steps && output_count < max_out) {
        out[output_count++] = DV_ERASURE;
        ctx->decided++;
      }
    } else {
      int frontier = dv_trellis_frontier(ctx, &d->trellises[win]);
      while (ctx->decided < ctx->steps && output_count < max_out) {
        out[output_count++] =
            dv_trellis_trace(ctx, &d->trellises[win], frontier, ctx->decided);
        ctx->decided++;
      }
    }
  }
  return output_count;
}
