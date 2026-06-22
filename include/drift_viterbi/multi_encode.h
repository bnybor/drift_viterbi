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

#ifndef INCLUDE_DRIFT_VITERBI_MULTI_ENCODE_H_
#define INCLUDE_DRIFT_VITERBI_MULTI_ENCODE_H_

#include <drift_viterbi/encode.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The sender's mirror of dv_multi_decoder: holds a family of codes that share a
 * rate and constraint length and one encoder state shared across them, and on
 * each call encodes with a code you pick by index. Opaque handle - make it with
 * dv_multi_encode_create(), free it with dv_multi_encode_destroy().
 */
typedef struct dv_multi_encoder dv_multi_encoder;

/* clang-format off */
/*
 * Settings for dv_multi_encode_create().
 *
 *   codes     : array of `codes_len` codes to encode with, selected per call by
 *               index. Required. They must share a rate (same dv_code_n) AND a
 *               constraint length (same dv_code_k): one shared encoder state
 *               drives whichever code is chosen, which is well defined precisely
 *               because a convolutional encoder's state transition depends only
 *               on the constraint length and the input bits, not the generator
 *               polynomials - so a same-K family steps one state in lockstep and
 *               typically differs only in its generators. A set that violates
 *               this is rejected (dv_multi_encode_create returns NULL). Each code
 *               must outlive the encoder; the array is copied, the codes are not.
 *   codes_len : how many codes `codes` points to.
 */
/* clang-format on */
typedef struct {
  const dv_code *const *codes;
  size_t codes_len;
} dv_multi_encode_params;

/*
 * Allocate a multi-encoder from `params` (must be non-NULL). The codes must share
 * a rate (dv_code_n) and a constraint length (dv_code_k), and each must outlive
 * the encoder (it copies the array and borrows the codes; it frees neither in
 * dv_multi_encode_destroy()). Returns NULL on a NULL params/codes, a NULL or
 * mismatched-rate/length code entry, or out of memory.
 */
dv_multi_encoder *dv_multi_encode_create(const dv_multi_encode_params *params);

/* Free a multi-encoder. The codes it borrowed are left untouched. NULL is fine. */
void dv_multi_encode_destroy(dv_multi_encoder *e);

/*
 * Encode `n_bits` input bits (each DV_FALSE or DV_TRUE) with codes[idx], writing
 * n_bits * dv_code_n bits to `out` (capacity `max_out`). The one encoder state is
 * shared across all the codes and advances here, so encoding is one continuous
 * stream across calls - and because the state is code-independent for a same-K
 * family, you may switch `idx` between calls to change codes mid-stream. Call
 * dv_multi_encode_flush() once the whole message is encoded.
 *
 * Returns the number of bits written, or DV_ERR_ARG (bad handle, idx out of
 * range, bad in/out arguments, or the output would not fit in max_out).
 */
int dv_multi_encode(dv_multi_encoder *e, int idx, const uint8_t *bits, int n_bits,
                    uint8_t *out, int max_out);

/*
 * Finish an encoded stream with codes[idx]: writes (dv_code_k - 1) * dv_code_n
 * trailing bits to `out` (capacity `max_out`) and returns the shared state to 0.
 * Mirror of dv_code_encode_flush. Returns the number of bits written, or
 * DV_ERR_ARG.
 */
int dv_multi_encode_flush(dv_multi_encoder *e, int idx, uint8_t *out, int max_out);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_DRIFT_VITERBI_MULTI_ENCODE_H_ */
