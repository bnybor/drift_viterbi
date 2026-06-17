# drift_viterbi

A small C library for **error correction on a bit stream**. You add redundancy
when you encode; on the other end you decode and get your bits back with errors
corrected — including bits that were **flipped, inserted, dropped, or lost**.
Inserted and dropped bits normally knock an error-correcting code out of sync;
this one stays aligned through them.

It works on a continuous stream: feed received bits in, read corrected bits out
at a fixed delay, with no message lengths or frame boundaries to manage.

## API

Pick a code, encode your bits, then stream-decode the received bits.

```c
#include "drift_viterbi/drift_viterbi.h"

/* 1. Pick a code (sender and receiver must use the same one). */
dv_code *code = dv_code_create_standard(DV_CODE_K7_RATE_1_2);
/* or define your own: dv_code_create(K, generators, num_generators) */

/* 2. Encode, in as many chunks as you like; carry `state` across calls. */
int state = 0, len = 0;
len += dv_code_encode(code, chunk1, n1, &state, out + len);
len += dv_code_encode(code, chunk2, n2, &state, out + len);
len += dv_code_encode_flush(code, &state, out + len);   /* finish the stream */

/* 3. Decode the received bits. */
dv_stream_decoder *d = dv_stream_decoder_create(code, &(dv_stream_params){
    .decision_depth = 40,   /* output delay; try ~6 * the code's K       */
    .max_drift      = 4,    /* set 0 to correct flips only                */
    .p_sub = 0.01, .p_ins = 0.01, .p_del = 0.01,
});

uint8_t decoded[OUT];
int n = dv_stream_decode(d, in, n_in, decoded, OUT);   /* feed + collect bits */
/* ... repeat as more bits arrive ... */
while (dv_stream_decode_flush(d, decoded, OUT) > 0)    /* drain at the end */
    /* use the decoded bits */ ;

dv_stream_decoder_destroy(d);
dv_code_destroy(code);
```

`dv_code` and `dv_stream_decoder` are opaque handles (`_create` / `_destroy`).
Functions return `DV_OK` (0) or a count on success, or a negative `DV_ERR_*`
code. Each bit is `DV_FALSE` or `DV_TRUE`, or `DV_ERASURE` to mark a received bit
as lost. `dv_code_n()` gives output-bits-per-input-bit for sizing encode buffers.

The decoder starts producing output after a short delay (`decision_depth` bits),
and it locks on whether you decode from the start of a stream or join one already
in progress — so discard the first ~`decision_depth` decoded bits (or send a
known preamble you can skip).

## Decoder settings

Set these in `dv_stream_params`; anything you leave out defaults to 0.

| Field            | What it does |
|------------------|--------------|
| `decision_depth` | Output delay in bits. Larger = more reliable, more latency. Try ~6× the code's K. Required. |
| `p_sub`          | How often a received bit is flipped (e.g. 0.01 = 1%). Required. |
| `max_drift`      | How far alignment may slip from inserted/dropped bits. 0 (default) corrects flips only; 4–8 also handles insertions and deletions. |
| `p_ins`, `p_del` | How often a bit is inserted / dropped. Needed when `max_drift > 0`. |
| `p_erase`        | How often a received bit is marked `DV_ERASURE` (lost). |

You don't need exact probabilities — rough, order-of-magnitude values are fine;
only their relative sizes matter. They mainly control how readily the decoder
assumes an inserted/dropped bit versus a plain flipped one:

- **Insert/drop probability too low:** the decoder resists realigning and tries
  to explain a real inserted/dropped bit as a run of flipped bits. A badly missed
  one can throw it out of sync for a stretch. Lean this way only if inserts and
  drops are genuinely rare.
- **Insert/drop probability too high:** the decoder over-corrects, imagining
  inserts/drops to explain ordinary noise and adding a few extra errors. This
  degrades gently.

When unsure, tune on representative data rather than chasing exact numbers.

## Build

```sh
cmake -S . -B build
cmake --build build
```

## Test

```sh
ctest --test-dir build --output-on-failure
```

## Install

```sh
cmake --install build --prefix /your/prefix
```

## License

MIT — see [LICENSE](LICENSE).
