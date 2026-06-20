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

#ifndef DRIFT_VITERBI_COMPARE_H
#define DRIFT_VITERBI_COMPARE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* clang-format off */
/*
 * drift_viterbi compare - decide, blindly, whether two received bit-streams were
 * produced by convolutional codes with the *same* generator polynomials. Unlike
 * the encoder and decoder this needs neither the code nor the transmitted bits:
 * it recovers each stream's parity-check structure and tests one against the
 * other, tolerating substitution noise and both constant and cumulative framing
 * drift.
 *
 *   double p = dv_compare(n, k, lhs, lhs_len, rhs, rhs_len);
 *
 * Bits follow the drift_viterbi convention: one bit per byte (the low bit of
 * each byte).
 */
/* clang-format on */

/*
 * Probability that the buffer `sample` (of `len` bits) carries some rate-1/n,
 * constraint-length-k convolutional code: ~1 when it does, ~0 when it does not.
 * This only asks whether a code is present, not which one - to test whether two
 * buffers share the same code, use dv_compare. Like dv_compare it recovers the
 * code's structure blindly and tolerates substitution noise, erasures, and both
 * constant and cumulative insertion/deletion drift. Returns a negative value
 * when the result cannot be determined (null buffer, too little data, or a code
 * whose relation window n*(k+1) exceeds 32).
 */
double dv_detect(int n, int k, uint8_t *sample, size_t len);

/*
 * The range of buffer lengths (in bits; one bit per byte) dv_detect can use for
 * a given rate 1/n and constraint length k - the same meaning as the dv_compare
 * pair below, applied to dv_detect's single buffer: dv_detect_min_len is the
 * fewest bits needed before detection is attempted (fewer is undetermined), and
 * dv_detect_max_len is the most bits it consults (a longer buffer may be
 * truncated). Both return a negative value when (n, k) is out of range.
 */
long dv_detect_min_len(int n, int k);
long dv_detect_max_len(int n, int k);

/*
 * Probability that, for a given rate 1/n and constraint length k, the two
 * stream samples have consistent convolutional-code generator polynomials: ~1
 * when they share the code, ~0 when they do not. Works on streams of any length
 * that is long enough to pin the code down, and on any code whose relation
 * window n*(k+1) fits a 32-bit word. Returns a negative value when the result
 * cannot be determined (bad arguments, too little data, or a code with n*(k+1)
 * > 32).
 */
double dv_compare(int n, int k, uint8_t *lhs, size_t lhs_len, uint8_t *rhs,
                  size_t rhs_len);

/* clang-format off */
/*
 * The range of sample lengths (in bits; one bit per byte, so this is also the
 * byte count) that dv_compare can use for a given rate 1/n and constraint
 * length k:
 *
 *   dv_compare_min_len - the fewest bits each sample must have for dv_compare to
 *       attempt a comparison; with fewer it returns DV_UNDETERMINED for want of
 *       data. (Necessary, not always sufficient: very degenerate data this long
 *       may still be undetermined.)
 *   dv_compare_max_len - the most bits dv_compare consults; beyond this, extra
 *       bits cannot change the result, so a longer sample may be truncated.
 *
 * Both return a negative value when (n, k) is out of range - the same codes for
 * which dv_compare cannot run (n < 1, k < 2, k > 9, or window n*(k+1) > 32).
 */
/* clang-format on */
long dv_compare_min_len(int n, int k);
long dv_compare_max_len(int n, int k);

#ifdef __cplusplus
}
#endif

#endif /* DRIFT_VITERBI_COMPARE_H */
