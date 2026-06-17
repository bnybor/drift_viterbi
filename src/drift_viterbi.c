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

#include "drift_viterbi/drift_viterbi.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct dv_code {
  int K;                    /* constraint length                   */
  int n;                    /* output bits per input bit           */
  int n_states;             /* 1 << (K-1)                          */
  unsigned int input_tap;   /* 1 << (K-1)                          */
  unsigned int *generators; /* [n]                                 */
  int *next_state;          /* [n_states*2], indexed [s*2 + b]     */
  uint8_t *output;          /* [n_states*2*n], [((s*2)+b)*n + j]   */
};

const char *drift_viterbi_version(void) { return "0.1.0"; }

/* Parity (XOR of all bits) of x: 1 if x has an odd number of set bits. */
static int parity(unsigned int x) {
  /* Fold down to a single parity bit. */
  x ^= x >> 16;
  x ^= x >> 8;
  x ^= x >> 4;
  x ^= x >> 2;
  x ^= x >> 1;
  return (int)(x & 1u);
}

/* ------------------------------------------------------------------------- */
/* Code and encoder                                                          */
/* ------------------------------------------------------------------------- */

dv_code *dv_code_create(int K, const unsigned int *generators,
                        int num_generators) {
  if (!generators || K < 2 || num_generators < 1) {
    return NULL;
  }

  dv_code *code = calloc(1, sizeof(*code));
  if (!code) {
    return NULL;
  }
  code->K = K;
  code->n = num_generators;
  code->n_states = 1 << (K - 1);
  code->input_tap = 1u << (K - 1);

  code->generators = malloc((size_t)code->n * sizeof(unsigned int));
  code->next_state = malloc((size_t)code->n_states * 2 * sizeof(int));
  code->output = malloc((size_t)code->n_states * 2 * code->n * sizeof(uint8_t));
  if (!code->generators || !code->next_state || !code->output) {
    dv_code_destroy(code);
    return NULL;
  }

  for (int g = 0; g < code->n; ++g) {
    code->generators[g] = generators[g];
  }

  /* Precompute the trellis: (state, input bit) -> (next_state, output). */
  for (int s = 0; s < code->n_states; ++s) {
    for (int b = 0; b <= 1; ++b) {
      unsigned int reg = ((unsigned int)b << (K - 1)) | (unsigned int)s;
      uint8_t *out = &code->output[((size_t)(s * 2 + b)) * code->n];
      for (int j = 0; j < code->n; ++j) {
        out[j] = (uint8_t)parity(reg & code->generators[j]);
      }
      int ns = ((s >> 1) | (b << (K - 2))) & (code->n_states - 1);
      code->next_state[s * 2 + b] = ns;
    }
  }

  return code;
}

dv_code *dv_code_create_standard(dv_standard_code which) {
  switch (which) {
    case DV_CODE_K3_RATE_1_2: {
      static const unsigned int g[] = {007, 005};
      return dv_code_create(3, g, 2);
    }
    case DV_CODE_K7_RATE_1_2: {
      static const unsigned int g[] = {0171, 0133};
      return dv_code_create(7, g, 2);
    }
    case DV_CODE_K7_RATE_1_3: {
      static const unsigned int g[] = {0171, 0165, 0133};
      return dv_code_create(7, g, 3);
    }
    case DV_CODE_K5_RATE_1_5: {
      static const unsigned int g[] = {037, 033, 025, 027, 035};
      return dv_code_create(5, g, 5);
    }
  }
  return NULL;
}

void dv_code_destroy(dv_code *code) {
  if (!code) {
    return;
  }
  free(code->generators);
  free(code->next_state);
  free(code->output);
  free(code);
}

int dv_code_n(const dv_code *code) { return code ? code->n : -1; }

int dv_code_k(const dv_code *code) { return code ? code->K : -1; }

/* Encode one input bit `b` from state *s: copy the code's n output bits to
 * `out`, advance *s, and return the number of bits written (the code's n). */
static int emit_group(const dv_code *code, int *s, int b, uint8_t *out) {
  const uint8_t *group = &code->output[((size_t)(*s * 2 + b)) * code->n];
  for (int j = 0; j < code->n; ++j) {
    out[j] = group[j];
  }
  *s = code->next_state[*s * 2 + b];
  return code->n;
}

int dv_code_encode(const dv_code *code, const uint8_t *bits, int n_bits,
                   int *state, uint8_t *out) {
  if (!code || !state || n_bits < 0 || (n_bits > 0 && !bits) || !out) {
    return DV_ERR_ARG;
  }
  if (*state < 0 || *state >= code->n_states) {
    return DV_ERR_ARG;
  }

  int s = *state, written = 0;
  for (int i = 0; i < n_bits; ++i) {
    written += emit_group(code, &s, bits[i] & 1, out + written);
  }
  *state = s;
  return written;
}

int dv_code_encode_flush(const dv_code *code, int *state, uint8_t *out) {
  if (!code || !state || !out) {
    return DV_ERR_ARG;
  }
  if (*state < 0 || *state >= code->n_states) {
    return DV_ERR_ARG;
  }

  /* Feed K-1 zero bits, which shift the state register back to 0. */
  int s = *state, written = 0;
  for (int i = 0; i < code->K - 1; ++i) {
    written += emit_group(code, &s, DV_FALSE, out + written);
  }
  *state = s;
  return written;
}

/* ------------------------------------------------------------------------- */
/* Streaming (sliding-window) decoder                                        */
/* ------------------------------------------------------------------------- */

/*
 * Runs the forward add-compare-select pass continuously over a buffer of
 * received bits, keeping only `L = decision_depth` steps of backpointers. A bit
 * is committed L steps after it is first seen, by which point the candidate
 * paths have merged. Output emerges at fixed latency L with bounded memory and
 * no frame boundaries.
 *
 * Re-anchoring keeps the drift window centred on the committed timing: each
 * step the window may be shifted by sigma in {-1,0,+1} (folded into the read
 * cursor), so the net cumulative drift can grow without bound while each node's
 * stored drift stays inside +/- max_drift. The shift is recorded per step so
 * traceback can translate node indices across the moving coordinate frame.
 */

/* Backpointer for one node: where it came from (prev_s, prev_di) and the input
 * bit that got there. */
typedef struct {
  int prev_s;
  int prev_di;
  unsigned char bit;
} dv_bp;

struct dv_stream_decoder {
  const dv_code *code; /* borrowed                            */
  int n, D, S, W, L;
  /* Branch-metric constants, in cost (negative-log-likelihood) units. */
  double c_match, c_miss, c_erase, c_keep, c_ins, c_del;

  double *metric;  /* [S*W], node-at-current-time costs            */
  double *nmetric; /* [S*W], scratch                               */
  dv_bp *bp;       /* [L*S*W], backpointer ring (layer = step % L) */
  int *shift;      /* [L], re-anchor sigma applied at each step    */

  long long steps;   /* trellis steps processed                      */
  long long decided; /* decisions emitted (next step index to emit)  */

  /* Received-bit buffer: valid bits live in rbuf[0 .. rlen). rpos is the
   * buffer index of the current step's zero-drift read base. */
  uint8_t *rbuf;
  int rcap, rlen, rpos;
};

/* Flat index into the [S][W] metric and backpointer arrays for the node at
 * encoder state s and drift index di (0..W-1; di == D means zero drift). */
static size_t node_at(int s, int di, int W) { return (size_t)s * W + di; }

/* -- received buffer ------------------------------------------------------- */

/* Drop the dead prefix. Keep 2*D bits of history below rpos so a re-anchor
 * that steps the cursor back still has its window buffered. */
static void compact_recv(struct dv_stream_decoder *sd) {
  int keep_from = sd->rpos - 2 * sd->D;
  if (keep_from <= 0) {
    return;
  }
  memmove(sd->rbuf, sd->rbuf + keep_from, (size_t)(sd->rlen - keep_from));
  sd->rlen -= keep_from;
  sd->rpos -= keep_from;
}

/* Ensure room for `extra` more bits, compacting then growing as needed. */
static int reserve_recv(struct dv_stream_decoder *sd, int extra) {
  compact_recv(sd);
  if (sd->rlen + extra > sd->rcap) {
    int ncap = sd->rcap * 2;
    if (ncap < sd->rlen + extra) {
      ncap = sd->rlen + extra;
    }
    uint8_t *nb = realloc(sd->rbuf, (size_t)ncap);
    if (!nb) {
      return DV_ERR_ALLOC;
    }
    sd->rbuf = nb;
    sd->rcap = ncap;
  }
  return DV_OK;
}

/* -- core trellis ---------------------------------------------------------- */

/* Flat index of the lowest-cost node at the current frontier (the node states
 * one step past the last one processed). */
static int frontier_best(const struct dv_stream_decoder *sd) {
  const size_t count = (size_t)sd->S * sd->W;
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
  if (sd->D == 0) {
    return 0; /* no drift tracking: window is one wide, nothing to shift */
  }
  const int best_di = frontier_best(sd) % sd->W;
  const int drift = best_di - sd->D;
  const int deadband = (sd->D + 1) / 2;
  if (drift >= deadband) {
    return +1;
  }
  if (drift <= -deadband) {
    return -1;
  }
  return 0;
}

/* Cost of reading the received group `window` (n bits) as the expected output
 * `expected`: an erased bit costs a fixed amount, others cost match or miss. */
static double branch_cost(const struct dv_stream_decoder *sd,
                          const uint8_t *expected, const uint8_t *window) {
  double cost = 0.0;
  for (int j = 0; j < sd->n; ++j) {
    if (window[j] == DV_ERASURE) {
      cost += sd->c_erase;
    } else {
      cost += (expected[j] == window[j]) ? sd->c_match : sd->c_miss;
    }
  }
  return cost;
}

/* Subtract the lowest node cost from every node, so the best one sits at 0.
 * Over an unbounded stream this keeps the costs from growing without limit. */
static void normalize(double *metric, size_t count) {
  double lowest = INFINITY;
  for (size_t i = 0; i < count; ++i) {
    if (metric[i] < lowest) {
      lowest = metric[i];
    }
  }
  if (lowest != INFINITY && lowest > 0.0) {
    for (size_t i = 0; i < count; ++i) {
      if (metric[i] != INFINITY) {
        metric[i] -= lowest;
      }
    }
  }
}

/* Shift the drift window by `sigma` (each node's drift index di -> di - sigma)
 * and advance the read cursor to match, so the bits actually read don't change
 * but the live drift values re-centre on the current timing. Nodes shifted out
 * of the window are dropped. Uses nmetric as scratch. */
static void reanchor(struct dv_stream_decoder *sd, int sigma) {
  const int S = sd->S, W = sd->W;
  for (int s = 0; s < S; ++s) {
    for (int di = 0; di < W; ++di) {
      const int src = di + sigma;
      sd->nmetric[node_at(s, di, W)] =
          (src >= 0 && src < W) ? sd->metric[node_at(s, src, W)] : INFINITY;
    }
  }
  memcpy(sd->metric, sd->nmetric, (size_t)S * W * sizeof(double));
  sd->rpos += sigma;
}

/* Advance one trellis step: re-anchor if needed, run the forward pass from
 * sd->metric into sd->nmetric (recording backpointers), normalise, and swap.
 * Branches that would read outside the buffered window are skipped, which is
 * what makes the ends of the stream safe. */
static void stream_step(struct dv_stream_decoder *sd) {
  const dv_code *code = sd->code;
  const int n = sd->n, D = sd->D, S = sd->S, W = sd->W;
  const size_t count = (size_t)S * W;
  const int delta[3] = {0, +1, -1}; /* keep / insert / delete */
  const double delta_cost[3] = {sd->c_keep, sd->c_ins, sd->c_del};

  const int sigma = pick_shift(sd);
  if (sigma != 0) {
    reanchor(sd, sigma);
  }
  sd->shift[sd->steps % sd->L] = sigma;

  for (size_t i = 0; i < count; ++i) {
    sd->nmetric[i] = INFINITY;
  }
  dv_bp *bp = sd->bp + (size_t)(sd->steps % sd->L) * count;

  /* For every live node, try both input bits and all three drift moves. */
  for (int s = 0; s < S; ++s) {
    for (int di = 0; di < W; ++di) {
      const double cur = sd->metric[node_at(s, di, W)];
      if (cur == INFINITY) {
        continue;
      }
      const int drift = di - D;
      const int read = sd->rpos + drift;
      if (read < 0 || read + n > sd->rlen) {
        continue; /* group not buffered yet */
      }
      const uint8_t *window = sd->rbuf + read;

      for (int b = 0; b <= 1; ++b) {
        const int next_s = code->next_state[s * 2 + b];
        const uint8_t *expected = &code->output[((size_t)(s * 2 + b)) * n];
        const double group_cost = branch_cost(sd, expected, window);

        for (int k = 0; k < 3; ++k) {
          const int next_drift = drift + delta[k];
          if (next_drift < -D || next_drift > D) {
            continue;
          }
          const double cost = cur + group_cost + delta_cost[k];
          const size_t to = node_at(next_s, next_drift + D, W);
          if (cost < sd->nmetric[to]) {
            sd->nmetric[to] = cost;
            bp[to].prev_s = s;
            bp[to].prev_di = di;
            bp[to].bit = (unsigned char)b;
          }
        }
      }
    }
  }

  normalize(sd->nmetric, count);

  double *tmp = sd->metric;
  sd->metric = sd->nmetric;
  sd->nmetric = tmp;
  sd->steps++;
  sd->rpos += n;
}

/* Walk the backpointers from frontier node `frontier` back to step `target` and
 * return the input bit decided there. Each step's backpointer is stored in that
 * step's drift frame, so when stepping back across a re-anchor we shift the
 * predecessor's drift index by that step's recorded sigma. */
static unsigned char trace_bit(const struct dv_stream_decoder *sd, int frontier,
                               long long target) {
  const size_t count = (size_t)sd->S * sd->W;
  int node = frontier;
  unsigned char bit = 0;
  for (long long i = sd->steps - 1; i >= target; --i) {
    const dv_bp *layer = sd->bp + (size_t)(i % sd->L) * count;
    const dv_bp e = layer[node];
    if (i == target) {
      bit = e.bit;
      break;
    }
    node = e.prev_s * sd->W + (e.prev_di + sd->shift[i % sd->L]);
  }
  return bit;
}

/* Process steps, emitting forced decisions, until input/output limits hit.
 * `draining` relaxes the look-ahead requirement for end-of-stream. */
static int run(struct dv_stream_decoder *sd, uint8_t *out, int max_out,
               int draining) {
  int oc = 0;
  for (;;) {
    if (!draining) {
      /* +1 of slack covers a re-anchor stepping the cursor forward. */
      if (sd->rlen < sd->rpos + sd->n + sd->D + 1) {
        break; /* not enough look-ahead yet */
      }
    } else {
      if (sd->rlen - sd->rpos < sd->n) {
        break; /* less than one group left  */
      }
    }

    /* Processing the next step overwrites the backpointer layer of step
     * (steps - L), so its decision must be emitted first. */
    if (sd->steps >= sd->L) {
      if (oc >= max_out) {
        break;
      }
      out[oc++] = trace_bit(sd, frontier_best(sd), sd->decided);
      sd->decided++;
    }
    stream_step(sd);
    if (sd->rpos - sd->D >= sd->rcap / 2) {
      compact_recv(sd);
    }
  }
  return oc;
}

/* Initialise the trellis metrics for blind acquisition: every encoder state is
 * equally likely (all at zero drift), so the decoder locks on whether it starts
 * at the stream's beginning or is tapped partway through. */
static void init_metric(struct dv_stream_decoder *sd) {
  const size_t count = (size_t)sd->S * sd->W;
  for (size_t i = 0; i < count; ++i) {
    sd->metric[i] = INFINITY;
  }
  for (int s = 0; s < sd->S; ++s) {
    sd->metric[node_at(s, sd->D, sd->W)] = 0.0; /* zero drift, cost 0 */
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
  sd->D = max_drift;
  sd->S = code->n_states;
  sd->W = 2 * max_drift + 1;
  sd->L = decision_depth;

  /* Channel model: a coded bit is erased with prob p_erase; otherwise it is
   * received and flipped with prob p_sub. The common (1 - p_erase) factor is
   * kept explicit so paths reading different erasure counts compare correctly
   * (p_erase = 0 reduces these to the plain hard-decision metric). */
  sd->c_match = -log((1.0 - p_erase) * (1.0 - p_sub));
  sd->c_miss = -log((1.0 - p_erase) * p_sub);
  sd->c_erase = -log(p_erase); /* +inf when p_erase == 0 (never read) */
  sd->c_keep = -log(1.0 - p_ins - p_del);
  sd->c_ins = -log(p_ins);
  sd->c_del = -log(p_del);

  const size_t count = (size_t)sd->S * sd->W;
  sd->metric = malloc(count * sizeof(double));
  sd->nmetric = malloc(count * sizeof(double));
  sd->bp = malloc((size_t)sd->L * count * sizeof(dv_bp));
  sd->shift = malloc((size_t)sd->L * sizeof(int));
  sd->rcap = (sd->L + 2) * sd->n + 8 * sd->D + 64;
  sd->rbuf = malloc((size_t)sd->rcap);
  if (!sd->metric || !sd->nmetric || !sd->bp || !sd->shift || !sd->rbuf) {
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
  free(sd->nmetric);
  free(sd->bp);
  free(sd->shift);
  free(sd->rbuf);
  free(sd);
}

int dv_stream_decode(dv_stream_decoder *sd, const uint8_t *in, int n_in,
                     uint8_t *out, int max_out) {
  if (!sd || (n_in > 0 && !in) || n_in < 0 || (max_out > 0 && !out) ||
      max_out < 0) {
    return DV_ERR_ARG;
  }
  int rc = reserve_recv(sd, n_in);
  if (rc < 0) {
    return rc;
  }
  memcpy(sd->rbuf + sd->rlen, in, (size_t)n_in);
  sd->rlen += n_in;
  return run(sd, out, max_out, /*draining=*/0);
}

int dv_stream_decode_flush(dv_stream_decoder *sd, uint8_t *out, int max_out) {
  if (!sd || (max_out > 0 && !out) || max_out < 0) {
    return DV_ERR_ARG;
  }
  int oc = run(sd, out, max_out, /*draining=*/1);

  /* Drain the pipeline: decide the remaining buffered steps from the final
   * frontier (reduced traceback depth for the last <= L bits). */
  if (sd->decided < sd->steps && oc < max_out) {
    int fb = frontier_best(sd);
    while (sd->decided < sd->steps && oc < max_out) {
      out[oc++] = trace_bit(sd, fb, sd->decided);
      sd->decided++;
    }
  }
  return oc;
}
