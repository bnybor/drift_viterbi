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

#ifndef DRIFT_VITERBI_DECODE_H
#define DRIFT_VITERBI_DECODE_H

/* The decoder is built from a dv_code, and shares the result codes and bit
 * values defined alongside the encoder. */
#include <drift_viterbi/encode.h>

#ifdef __cplusplus
extern "C" {
#endif

/* clang-format off */
/*
 * The receiver's half of drift_viterbi (the sender's half is in encode.h).
 *
 * Recover your original bits from a received stream: feed received bits to a
 * decoder and read your bits back, with flipped, inserted and dropped bits
 * corrected.
 *
 *   dv_stream_decoder *d = dv_stream_decoder_create(code, &(dv_stream_params){
 *       .decision_depth = 40,
 *       .max_drift      = 4,
 *       .p_sub = 0.01, .p_ins = 0.01, .p_del = 0.01,
 *   });
 *   int n = dv_stream_decode(d, in, n_in, out, NULL, out_cap);
 *   while (dv_stream_decode_flush(d, out, out_cap) > 0) { }
 *
 *   dv_stream_decoder_destroy(d);
 *
 * `code` must be the same one the sender used, and must stay alive until the
 * decoder is freed. dv_stream_decoder is an opaque handle.
 */
/* clang-format on */

/*
 * In received data you may mark a bit DV_ERASURE to say "this one was lost"; the
 * decoder then treats it as unknown instead of guessing 0 or 1. Ordinary bits
 * are DV_FALSE or DV_TRUE (see encode.h).
 */
#define DV_ERASURE ((uint8_t)0xFFu)

/* ------------------------------------------------------------------------- */
/* Decoder                                                                   */
/* ------------------------------------------------------------------------- */

/*
 * Recovers your original bits from a received stream: corrects flipped bits and
 * keeps its place through inserted or dropped ones. Push received bits in, pull
 * decoded bits out, with a fixed delay.
 *
 * You may start at the beginning of a stream or join one mid-flight; either way
 * the first ~decision_depth decoded bits come out while the decoder is still
 * locking on, so discard them (or send a known preamble you can skip). Opaque
 * handle.
 */
typedef struct dv_stream_decoder dv_stream_decoder;

/* clang-format off */
/*
 * Decoder settings for dv_stream_decoder_create(). Use designated initializers;
 * any field you leave out is 0.
 *
 *   decision_depth : output delay, in bits, before each bit is committed. Bigger
 *                    is more reliable but slower to emit. Try ~6 * dv_code_k().
 *                    Required (must be >= 1).
 *   p_sub          : how often a received bit is flipped, 0 < p_sub < 1 (e.g.
 *                    0.01 for 1%). Required.
 *   max_drift      : how far alignment may slip from inserted/dropped bits before
 *                    the decoder loses track. 0 (the default) corrects flipped
 *                    bits only; 4-8 also recovers from insertions and deletions.
 *   p_ins, p_del   : how often a coded bit is inserted / dropped, per bit and at
 *                    any position (p_ins + p_del < 1). Required when
 *                    max_drift > 0; leave 0 otherwise.
 *   p_erase        : how often a received bit is DV_ERASURE. 0 (the default) if
 *                    you never mark erasures.
 *
 * Rough probabilities are fine; only their relative sizes matter.
 */
/* clang-format on */
typedef struct {
  int decision_depth;
  int max_drift;
  double p_sub;
  double p_ins;
  double p_del;
  double p_erase;
} dv_stream_params;

/*
 * Make a decoder for `code` (which must stay alive until the decoder is freed)
 * using `params`. Returns NULL on invalid settings or out of memory; free it
 * with dv_stream_decoder_destroy().
 */
dv_stream_decoder *dv_stream_decoder_create(const dv_code *code,
                                            const dv_stream_params *params);

/* Free a decoder. Passing NULL is fine. */
void dv_stream_decoder_destroy(dv_stream_decoder *d);

/*
 * Feed `n_in` received bits (each DV_FALSE, DV_TRUE, or DV_ERASURE) and collect
 * up to `max_out` decoded bits into `out`. Returns how many decoded bits were
 * written (0 or more), or a negative DV_ERR_* code.
 *
 * You get about one decoded bit per dv_code_n(code) received bits. If `out`
 * fills up (return value == max_out), call again to collect more before feeding
 * more input.
 *
 * `lock_probability`, which may be NULL, stores the probability at each bit
 * position that the decoder is locked onto a valid coded bit stream.  Same size
 * as `out`.
 */
int dv_stream_decode(dv_stream_decoder *d, const uint8_t *in, int n_in,
                     uint8_t *out, double *lock_probability, int max_out);

/*
 * Call at the end of the stream to get the last decoded bits still in flight.
 * Returns how many bits were written (0..max_out); call it repeatedly until it
 * returns 0, after which every bit has been decoded.
 */
int dv_stream_decode_flush(dv_stream_decoder *d, uint8_t *out, int max_out);

#ifdef __cplusplus
}
#endif

#endif /* DRIFT_VITERBI_DECODE_H */
