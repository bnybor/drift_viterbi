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

#ifndef INCLUDE_DRIFT_VITERBI_MULTI_DECODE_H_
#define INCLUDE_DRIFT_VITERBI_MULTI_DECODE_H_

#include <drift_viterbi/decode.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Runs several stream decoders over one received stream and, per output bit,
 * keeps the bit from whichever is most confidently locked. Opaque handle - make
 * it with dv_multi_create(), free it with dv_multi_destroy().
 */
typedef struct dv_multi_decoder dv_multi_decoder;

/* clang-format off */
/*
 * Settings for dv_multi_create(). Use designated initializers; any tuning field
 * left 0 takes its default.
 *
 *   codes      : array of `codes_len` codes to decode against in parallel over a
 *                single shared received buffer and cadence. Required. They must
 *                share a rate (same dv_code_n) AND a constraint length (same
 *                dv_code_k) - one decoded bit corresponds to one message bit
 *                across all of them, and they advance in lockstep under one
 *                shared re-anchor over one trellis geometry - and otherwise
 *                typically differ only in their generator polynomials. A set
 *                that violates this is rejected (dv_multi_create returns NULL).
 *                Each code must outlive the multi-decoder.
 *   codes_len  : how many codes `codes` points to.
 *   stream     : decoder settings (decision_depth, drift, channel probabilities;
 *                see dv_stream_params) shared by every code's decoder. Required.
 *   lock_floor : the lock probability (see dv_stream_decode) the best-fitting
 *                code must reach for any bit to be committed; if none does,
 *                nothing is locked and the bit is DV_ERASURE. Default 0.6.
 *   lock_margin: how decisively the codes' likelihood-weighted vote must favour
 *                one bit value over the other - as a share of the total weight,
 *                so 0 commits on any majority and 1 never commits - for that bit
 *                to be emitted rather than DV_ERASURE. Default 0.2.
 */
/* clang-format on */
typedef struct {
  const dv_code *const *codes;
  size_t codes_len;
  dv_stream_params stream;
  double lock_floor;
  double lock_margin;
} dv_multi_params;

/*
 * Allocate a multi-decoder from `params` (must be non-NULL), building one stream
 * decoder per code in params->codes using params->stream. The codes must outlive
 * the multi-decoder; it owns the decoders it builds and frees them in
 * dv_multi_destroy(). Returns NULL on a NULL params, invalid settings, or out of
 * memory.
 */
dv_multi_decoder *dv_multi_create(const dv_multi_params *params);

/* Free a multi-decoder and every stream decoder it owns. Passing NULL is fine. */
void dv_multi_destroy(dv_multi_decoder *m);

/*
 * Decode a received bit stream with all of the decoders in the multi struct at
 * once. For each output-bit position the decoders' traced bits are combined by
 * likelihood weight - each code's weight falls off with how poorly it fits the
 * stream - and the heavier bit value is emitted when it leads the other by
 * lock_margin of the total weight. When no code is locked (none reaches
 * lock_floor) or the weighted vote is too close, that bit is DV_ERASURE.
 *
 * `in`/`n_in`, `out`/`max_out`, and the return value follow dv_stream_decode.
 * `locked_decoder`, which may be NULL, is written alongside `out` (same length):
 * each entry is the index of the likeliest code (lowest-cost) at that position,
 * or -1 where the bit was erased.
 */
int dv_multi_decode(dv_multi_decoder *d, const uint8_t *in, int n_in,
                    uint8_t *out, int *locked_decoder, int max_out);

/*
 * Flush a multi-decoder at end of stream, draining the last bits still in
 * flight in every wrapped decoder. Like dv_stream_decode_flush: call repeatedly
 * until it returns 0, after which every bit has been decoded. These tail bits
 * carry no lock probability to choose a decoder by, so they are decoded by the
 * decoder that was most recently locked during dv_multi_decode - or emitted as
 * DV_ERASURE if no decoder ever locked.
 */
int dv_multi_decode_flush(dv_multi_decoder *d, uint8_t *out, int max_out);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_DRIFT_VITERBI_MULTI_DECODE_H_ */
