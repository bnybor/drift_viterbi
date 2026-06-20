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
 * Tests for dv_detect: does a single buffer carry any code at (n, k)? Clean
 * coded data detects high; random data low; detection survives indels and
 * erasures, like dv_compare; too-short, out-of-range, or null inputs are
 * undetermined.
 */

#include "dv_test_util.h"

#define CODED_CAP MAX_CODED(4000)

static void test_detect(uint64_t seed) {
  printf("test_detect\n");
  const dv_standard_code codes[] = {DV_CODE_K3_RATE_1_2, DV_CODE_K7_RATE_1_2,
                                    DV_CODE_K7_RATE_1_3, DV_CODE_K5_RATE_1_5};
  const char *names[] = {"K3_R1_2", "K7_R1_2", "K7_R1_3", "K5_R1_5"};
  const int info_bits = 1500;

  uint8_t *msg = malloc((size_t)info_bits);
  uint8_t *coded = malloc(CODED_CAP);
  uint8_t *chan = malloc(CODED_CAP);

  for (int c = 0; c < 4; ++c) {
    uint64_t rng = seed + 700 + (uint64_t)c;
    int n, k;
    rand_bits(msg, info_bits, &rng);
    size_t clen = encode(codes[c], msg, info_bits, coded, &n, &k);

    /* Clean coded data detects strongly; a deletion channel (cumulative drift)
     * and a few percent erasures both stay above threshold. */
    char label[48];
    snprintf(label, sizeof label, "%s clean", names[c]);
    check_gt(label, dv_detect(n, k, coded, clen), 0.8);

    size_t dlen = delete_channel(coded, clen, 0.01, &rng, chan);
    snprintf(label, sizeof label, "%s indel", names[c]);
    check_gt(label, dv_detect(n, k, chan, dlen), 0.7);

    memcpy(chan, coded, clen);
    erase_channel(chan, clen, 0.04, &rng);
    snprintf(label, sizeof label, "%s erased", names[c]);
    check_gt(label, dv_detect(n, k, chan, clen), 0.7);
  }

  /* Random (non-coded) buffer -> low. */
  {
    uint64_t rng = seed + 800;
    rand_bits(coded, 3000, &rng);
    check_lt("random not detected", dv_detect(2, 7, coded, 3000), 0.3);
  }

  /* Too short, out-of-range, and null -> undetermined (negative). */
  check_undetermined("too-short", dv_detect(2, 7, coded, 8));
  check_undetermined("window>32", dv_detect(5, 9, coded, 3000));
  check_undetermined("null buffer", dv_detect(2, 7, NULL, 3000));

  free(msg);
  free(coded);
  free(chan);
}

/* Blind acquisition: detection must survive a capture that does not begin at a
 * clean point - spliced mid-stream, behind a garbage / partial-codeword prefix,
 * or at an arbitrary bit phase - mirroring the decoder's blind acquisition. A
 * garbage prefix on otherwise-random data must still read as no code. */
static void test_blind_acquisition_detect(uint64_t seed) {
  printf("test_blind_acquisition_detect\n");
  const int info_bits = 1500;
  uint8_t *msg = malloc((size_t)info_bits);
  uint8_t *coded = malloc(CODED_CAP);
  uint8_t *buf = malloc(CODED_CAP + 4096);

  uint64_t rng = seed + 900;
  int n, k;
  rand_bits(msg, info_bits, &rng);
  size_t clen = encode(DV_CODE_K7_RATE_1_2, msg, info_bits, coded, &n, &k);

  /* Spliced mid-stream: a tap that begins partway through the coded stream, at
   * an offset well past the framing-phase search and not a multiple of n. */
  const size_t splice = 137;
  check_gt("splice", dv_detect(n, k, coded + splice, clen - splice), 0.7);

  /* Leading garbage / partial-codeword prefixes of growing length. */
  const size_t prefixes[] = {1, 3, 5, 17, 64, 256, 1024};
  for (size_t i = 0; i < sizeof prefixes / sizeof *prefixes; ++i) {
    size_t blen = prepend_prefix(prefixes[i], coded, clen, &rng, buf);
    char label[48];
    snprintf(label, sizeof label, "garbage-prefix-%zu", prefixes[i]);
    check_gt(label, dv_detect(n, k, buf, blen), 0.7);
  }

  /* A garbage prefix on otherwise-random data is still not a code. */
  rand_bits(coded, (int)clen, &rng);
  size_t rlen = prepend_prefix(256, coded, clen, &rng, buf);
  check_lt("garbage-prefix-random", dv_detect(n, k, buf, rlen), 0.3);

  free(msg);
  free(coded);
  free(buf);
}

int main(void) {
  test_detect(0xD1F7C0DEULL);
  test_blind_acquisition_detect(0xD1F7C0DEULL);
  return test_summary("detect");
}
