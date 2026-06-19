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

#ifndef DRIFT_VITERBI_DRIFT_VITERBI_H
#define DRIFT_VITERBI_DRIFT_VITERBI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* clang-format off */
/*
 * drift_viterbi - forward error correction for a stream of bits sent over a
 * noisy channel. It corrects flipped bits, and - unlike most error-correcting
 * codes - also keeps its place when bits are accidentally inserted or dropped.
 *
 * Sender:   pick a code, then encode your bits (this adds redundancy).
 * Receiver: feed the received bits to a decoder and read your bits back, with
 *           errors corrected.
 *
 *   dv_code *code = dv_code_create_standard(DV_CODE_K7_RATE_1_2);
 *
 *   // encode (in one or more chunks)
 *   int state = 0;
 *   int len  = dv_code_encode(code, bits, n_bits, &state, out);
 *   len     += dv_code_encode_flush(code, &state, out + len);
 *
 *   // decode: feed received bits, collect decoded bits, drain at the end
 *   dv_stream_decoder *d = dv_stream_decoder_create(code, &(dv_stream_params){
 *       .decision_depth = 40,
 *       .max_drift      = 4,
 *       .p_sub = 0.01, .p_ins = 0.01, .p_del = 0.01,
 *   });
 *   int n = dv_stream_decode(d, in, n_in, out, out_cap);
 *   while (dv_stream_decode_flush(d, out, out_cap) > 0) { }
 *
 *   dv_stream_decoder_destroy(d);
 *   dv_code_destroy(code);
 *
 * dv_code and dv_stream_decoder are opaque handles - create and free them with
 * the matching functions. Functions return DV_OK (0) or a count on success, or
 * a negative DV_ERR_* code.
 */
/* clang-format on */

/* Result codes for functions that don't return a count. */
enum {
  DV_OK = 0,
  DV_ERR_ARG = -1,  /* a bad argument was passed */
  DV_ERR_ALLOC = -2 /* ran out of memory         */
};

/*
 * Every bit is DV_FALSE or DV_TRUE. In received data you may also mark a bit
 * DV_ERASURE to say "this one was lost"; the decoder then treats it as unknown
 * instead of guessing 0 or 1.
 */
#define DV_FALSE ((uint8_t)0u)
#define DV_TRUE ((uint8_t)1u)
#define DV_ERASURE ((uint8_t)0xFFu)

/* This library's version, e.g. "0.1.0". */
const char *drift_viterbi_version(void);

/* ------------------------------------------------------------------------- */
/* Code (the error-correction scheme)                                        */
/* ------------------------------------------------------------------------- */

/* A code is the redundancy scheme; sender and receiver must use the same one.
 * Opaque handle - make it below, free it with dv_code_destroy(). */
typedef struct dv_code dv_code;

/* Ready-made codes. More output per input bit corrects more errors but uses
 * more bandwidth. When unsure, pick DV_CODE_K7_RATE_1_2. */
typedef enum {
  DV_CODE_K3_RATE_1_2, /* 2x output size, light correction      */
  DV_CODE_K7_RATE_1_2, /* 2x output size, strong (good default) */
  DV_CODE_K7_RATE_1_3, /* 3x output size, stronger              */
  DV_CODE_K5_RATE_1_5  /* 5x output size, strongest             */
} dv_standard_code;

/* Make one of the ready-made codes above. Returns NULL on a bad argument or out
 * of memory; free the result with dv_code_destroy(). */
dv_code *dv_code_create_standard(dv_standard_code which);

/*
 * Make a custom code - most users want dv_code_create_standard() instead. `K`
 * is the code's memory (2..9); `generators` is an array of `num_generators` tap
 * masks of K bits each, and `num_generators` sets how many output bits each
 * input bit becomes. Returns NULL on a bad argument or out of memory.
 */
dv_code *dv_code_create(int K, const unsigned int *generators,
                        int num_generators);

/* Free a code. Passing NULL is fine. */
void dv_code_destroy(dv_code *code);

/* Output bits produced per input bit, so the encoded size of n_bits is
 * n_bits * dv_code_n(code). Returns -1 if code is NULL. */
int dv_code_n(const dv_code *code);

/* The code's memory length K. A good decision_depth is about 6 * K. Returns -1
 * if code is NULL. */
int dv_code_k(const dv_code *code);

/*
 * Encode `n_bits` input bits (each DV_FALSE or DV_TRUE) into `out`, which needs
 * room for n_bits * dv_code_n(code) bits.
 *
 * Encoding is one continuous stream: keep an `int state`, set it to 0 before
 * the first call, and pass the same variable to every call - so you can encode
 * in as many chunks as you like. When the whole message is encoded, call
 * dv_code_encode_flush() once to finish it.
 *
 * Returns the number of bits written, or DV_ERR_ARG.
 */
int dv_code_encode(const dv_code *code, const uint8_t *bits, int n_bits,
                   int *state, uint8_t *out);

/*
 * Finish an encoded stream: writes (K-1) * dv_code_n(code) trailing bits so the
 * decoder can recover the last input bits cleanly. Pass the same `state` you
 * gave dv_code_encode(). Returns the number of bits written, or DV_ERR_ARG.
 */
int dv_code_encode_flush(const dv_code *code, int *state, uint8_t *out);

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
 */
int dv_stream_decode(dv_stream_decoder *d, const uint8_t *in, int n_in,
                     uint8_t *out, int max_out);

/*
 * Call at the end of the stream to get the last decoded bits still in flight.
 * Returns how many bits were written (0..max_out); call it repeatedly until it
 * returns 0, after which every bit has been decoded.
 */
int dv_stream_decode_flush(dv_stream_decoder *d, uint8_t *out, int max_out);

#ifdef __cplusplus
}
#endif

#endif /* DRIFT_VITERBI_DRIFT_VITERBI_H */
