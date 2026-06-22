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

#include <drift_viterbi/encode.h>

#include <drift_viterbi/stdlib.h>

#include "dv_internal.h"

const char *drift_viterbi_version(void) { return "0.1.0"; }

/* Parity (XOR of all bits) of `bits`: 1 if it has an odd number of set bits. */
static int parity(unsigned int bits) {
  /* Fold down to a single parity bit. */
  bits ^= bits >> 16;
  bits ^= bits >> 8;
  bits ^= bits >> 4;
  bits ^= bits >> 2;
  bits ^= bits >> 1;
  return (int)(bits & 1u);
}

/* ------------------------------------------------------------------------- */
/* Code and encoder                                                          */
/* ------------------------------------------------------------------------- */

dv_code *dv_code_create(int K, const unsigned int *generators,
                        int num_generators) {
  /* K is documented as 2..9 (encode.h). The upper bound also keeps 1 << (K-1)
   * well clear of int-shift UB and the trellis a sane size. */
  if (!generators || K < 2 || K > 9 || num_generators < 1) {
    return NULL;
  }

  dv_code *code = dv_calloc(1, sizeof(*code));
  if (!code) {
    return NULL;
  }
  code->K = K;
  code->n = num_generators;
  code->n_states = 1 << (K - 1);
  code->input_tap = 1u << (K - 1);

  code->generators = dv_malloc((size_t)code->n * sizeof(unsigned int));
  code->next_state = dv_malloc((size_t)code->n_states * 2 * sizeof(int));
  code->output = dv_malloc((size_t)code->n_states * 2 * code->n * sizeof(uint8_t));
  if (!code->generators || !code->next_state || !code->output) {
    dv_code_destroy(code);
    return NULL;
  }

  for (int generator_index = 0; generator_index < code->n; ++generator_index) {
    code->generators[generator_index] = generators[generator_index];
  }

  /* Precompute the trellis: (state, input bit) -> (next_state, output). */
  for (int state = 0; state < code->n_states; ++state) {
    for (int bit = 0; bit <= 1; ++bit) {
      unsigned int shift_register =
          ((unsigned int)bit << (K - 1)) | (unsigned int)state;
      uint8_t *out = &code->output[((size_t)(state * 2 + bit)) * code->n];
      for (int j = 0; j < code->n; ++j) {
        out[j] = (uint8_t)parity(shift_register & code->generators[j]);
      }
      int next_state = ((state >> 1) | (bit << (K - 2))) & (code->n_states - 1);
      code->next_state[state * 2 + bit] = next_state;
    }
  }

  return code;
}

dv_code *dv_code_create_standard(dv_standard_code which) {
  switch (which) {
    /* These generator sets and the d_free values in encode.h are produced by
     * bench/dv_codesearch, which selects, per family, the codes that are
     * mutually distinguishable under the decoder's lock metric (rate-1/2 tops
     * out at three such codes; the wider rates reach five). */
    case DV_CODE_K3_RATE_1_2: {
      static const unsigned int generators[] = {005, 007};
      return dv_code_create(3, generators, 2);
    }
    case DV_CODE_K3_RATE_1_2_ALT1: {
      static const unsigned int generators[] = {001, 007};
      return dv_code_create(3, generators, 2);
    }
    case DV_CODE_K3_RATE_1_2_ALT2: {
      static const unsigned int generators[] = {003, 007};
      return dv_code_create(3, generators, 2);
    }
    case DV_CODE_K7_RATE_1_2: {
      static const unsigned int generators[] = {0171, 0133};
      return dv_code_create(7, generators, 2);
    }
    case DV_CODE_K7_RATE_1_2_ALT1: {
      static const unsigned int generators[] = {0043, 0175};
      return dv_code_create(7, generators, 2);
    }
    case DV_CODE_K7_RATE_1_2_ALT2: {
      static const unsigned int generators[] = {0107, 0156};
      return dv_code_create(7, generators, 2);
    }
    case DV_CODE_K7_RATE_1_3: {
      static const unsigned int generators[] = {0113, 0135, 0157};
      return dv_code_create(7, generators, 3);
    }
    case DV_CODE_K7_RATE_1_3_ALT1: {
      static const unsigned int generators[] = {0112, 0153, 0157};
      return dv_code_create(7, generators, 3);
    }
    case DV_CODE_K7_RATE_1_3_ALT2: {
      static const unsigned int generators[] = {0037, 0135, 0153};
      return dv_code_create(7, generators, 3);
    }
    case DV_CODE_K7_RATE_1_3_ALT3: {
      static const unsigned int generators[] = {0012, 0145, 0177};
      return dv_code_create(7, generators, 3);
    }
    case DV_CODE_K7_RATE_1_3_ALT4: {
      static const unsigned int generators[] = {0042, 0133, 0172};
      return dv_code_create(7, generators, 3);
    }
    case DV_CODE_K5_RATE_1_5: {
      static const unsigned int generators[] = {025, 027, 033, 035, 037};
      return dv_code_create(5, generators, 5);
    }
    case DV_CODE_K5_RATE_1_5_ALT1: {
      static const unsigned int generators[] = {007, 017, 025, 027, 035};
      return dv_code_create(5, generators, 5);
    }
    case DV_CODE_K5_RATE_1_5_ALT2: {
      static const unsigned int generators[] = {011, 032, 033, 035, 037};
      return dv_code_create(5, generators, 5);
    }
    case DV_CODE_K5_RATE_1_5_ALT3: {
      static const unsigned int generators[] = {013, 021, 023, 033, 037};
      return dv_code_create(5, generators, 5);
    }
    case DV_CODE_K5_RATE_1_5_ALT4: {
      static const unsigned int generators[] = {013, 024, 032, 033, 037};
      return dv_code_create(5, generators, 5);
    }
  }
  return NULL;
}

void dv_code_destroy(dv_code *code) {
  if (!code) {
    return;
  }
  dv_free(code->generators);
  dv_free(code->next_state);
  dv_free(code->output);
  dv_free(code);
}

int dv_code_n(const dv_code *code) { return code ? code->n : -1; }

int dv_code_k(const dv_code *code) { return code ? code->K : -1; }

/* Encode one input bit `bit` from state *state: copy the code's n output bits
 * to `out`, advance *state, and return the number of bits written (the code's
 * n). */
static int emit_group(const dv_code *code, int *state, int bit, uint8_t *out) {
  const uint8_t *group = &code->output[((size_t)(*state * 2 + bit)) * code->n];
  for (int j = 0; j < code->n; ++j) {
    out[j] = group[j];
  }
  *state = code->next_state[*state * 2 + bit];
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

  int current_state = *state, written = 0;
  for (int i = 0; i < n_bits; ++i) {
    written += emit_group(code, &current_state, bits[i] & 1, out + written);
  }
  *state = current_state;
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
  int current_state = *state, written = 0;
  for (int i = 0; i < code->K - 1; ++i) {
    written += emit_group(code, &current_state, DV_FALSE, out + written);
  }
  *state = current_state;
  return written;
}
