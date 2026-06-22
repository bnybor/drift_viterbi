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
 * Multi-encoder: the sender's mirror of the multi-decoder. It holds a family of
 * codes that share a rate and constraint length and ONE encoder state shared
 * across them, and on each call encodes with a code chosen by index. The shared
 * state is well defined because a convolutional encoder's state transition is a
 * function of the constraint length and the input bits only, not the generator
 * polynomials (see emit_group/next_state in encode.c): a same-K family steps the
 * single state in lockstep, so switching the chosen index between calls changes
 * the code mid-stream without breaking continuity - the encode-side image of the
 * multi-decoder following a code change.
 */

#include <drift_viterbi/multi_encode.h>

#include <drift_viterbi/encode.h>
#include <drift_viterbi/stdlib.h>

struct dv_multi_encoder {
  const dv_code **codes; /* [n] borrowed; each must outlive the encoder */
  size_t n;
  int state; /* one shared shift-register state for all codes */
};

dv_multi_encoder *dv_multi_encode_create(const dv_multi_encode_params *params) {
  if (!params || (params->codes_len > 0 && !params->codes)) {
    return NULL;
  }
  dv_multi_encoder *e = dv_calloc(1, sizeof(*e));
  if (!e) {
    return NULL;
  }
  e->n = params->codes_len;

  if (params->codes_len > 0) {
    /* calloc so a build failure partway leaves the rest NULL and
     * dv_multi_encode_destroy can free what was built. */
    e->codes = dv_calloc(params->codes_len, sizeof(const dv_code *));
    if (!e->codes) {
      dv_multi_encode_destroy(e);
      return NULL;
    }
    for (size_t j = 0; j < params->codes_len; ++j) {
      /* One shared state drives every code, so all codes must agree on the
       * constraint length (dv_code_k) it steps and the rate (dv_code_n) each
       * step emits; otherwise the chosen code's stream would not be coherent
       * with the others'. A NULL slot fails here too: dv_code_n(NULL) == -1 !=
       * codes[0]'s rate. Mirrors dv_multi_create's gate. */
      if (dv_code_n(params->codes[j]) != dv_code_n(params->codes[0]) ||
          dv_code_k(params->codes[j]) != dv_code_k(params->codes[0])) {
        dv_multi_encode_destroy(e);
        return NULL;
      }
      e->codes[j] = params->codes[j];
    }
  }
  return e;
}

void dv_multi_encode_destroy(dv_multi_encoder *e) {
  if (!e) {
    return;
  }
  dv_free(e->codes);
  dv_free(e);
}

/* Common argument check for the two public entry points; mirrors dv_code_encode's
 * contract plus the bounds-safety contract of dv_multi_decode (max_out). */
static int encode_args_ok(const dv_multi_encoder *e, int idx, const uint8_t *out,
                          int max_out) {
  return e && idx >= 0 && (size_t)idx < e->n && !(max_out > 0 && !out) &&
         max_out >= 0;
}

int dv_multi_encode(dv_multi_encoder *e, int idx, const uint8_t *bits, int n_bits,
                    uint8_t *out, int max_out) {
  if (!encode_args_ok(e, idx, out, max_out) || n_bits < 0 ||
      (n_bits > 0 && !bits)) {
    return DV_ERR_ARG;
  }
  if (n_bits * dv_code_n(e->codes[idx]) > max_out) {
    return DV_ERR_ARG;
  }
  return dv_code_encode(e->codes[idx], bits, n_bits, &e->state, out);
}

int dv_multi_encode_flush(dv_multi_encoder *e, int idx, uint8_t *out,
                          int max_out) {
  if (!encode_args_ok(e, idx, out, max_out)) {
    return DV_ERR_ARG;
  }
  if ((dv_code_k(e->codes[idx]) - 1) * dv_code_n(e->codes[idx]) > max_out) {
    return DV_ERR_ARG;
  }
  return dv_code_encode_flush(e->codes[idx], &e->state, out);
}
