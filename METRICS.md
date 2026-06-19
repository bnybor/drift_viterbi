# drift_viterbi metrics

`bench/dv_metrics.c` measures the decoding-mistake rate as a function of the
channel's flip / insert / delete / erase rates, for all four standard codes. It
runs a Monte-Carlo sweep — random message → encode → channel → stream-decode —
and reports the **normalized edit (Levenshtein) distance** between the decoded
bits and the original message, divided by the number of message bits (after a
warm-up to skip the acquisition transient). Each axis is swept independently with
the other rates at zero, so each curve isolates one impairment.

> Edit distance is the right metric for a channel that inserts and deletes bits:
> a single uncorrected sync slip costs *one* edit, rather than misaligning the
> whole remaining stream the way a position-by-position bit comparison would. A
> clean decode scores 0; total failure approaches ~1 edit per info bit.

```sh
# Build the harness (off by default) and run the sweep to CSV.
cmake -S . -B build -DDRIFT_VITERBI_BUILD_BENCH=ON
cmake --build build --target dv_metrics
build/bench/dv_metrics > metrics.csv          # defaults: 12 trials, 4000 info bits
# build/bench/dv_metrics <trials> <info_bits> <seed>   # e.g. 3 1000 1 for a quick run

# Plot the mistake metric vs each channel rate (one curve per code). Needs matplotlib:
python3 -m venv .venv && .venv/bin/pip install matplotlib
.venv/bin/python bench/plot_metrics.py metrics.csv -o plots/
```

The default sweep takes a few minutes (the drift-tracking axes dominate); pass
smaller `trials`/`info_bits` for a faster, coarser run. The sweep is fanned out
across cores with OpenMP when available, and each point owns a seeded PRNG
stream, so a given `seed` reproduces exactly regardless of thread count.

By default the plotter writes, per axis, two metrics × two units —
`plots/{edit,runlen}_vs_{flip,insert,delete,erase}_per_{info,coded}_bit.png`:

- **edit** — edit distance per bit (mistakes per bit).
- **runlen** — mean run length between edits (`1 / edit_rate`): the average bits
  that get through between mistakes, i.e. how long a transmission you can expect
  to push through cleanly. Points where no edits were observed are dropped (the
  value there is only a lower bound, not infinity).

Since the channel rate (x-axis) is per coded bit, the per-coded-bit view divides
by the code's rate `n` so codes of different rates compare fairly. Run-length
plots are linear with an adaptive y-cap (the low-rate spikes run off the top);
pass `--logy` for a log axis or `--ymax N` to set the cap. Pass
`--metric edit|runlength` or `--unit info|coded` to emit just one.

## Generated plots

The figures below come from a 30-trial × 4000-info-bit sweep (`dv_metrics 30
4000`, default seed). In every plot the x-axis is the channel impairment rate
per coded bit, and the four curves are the standard codes — `K3_R1_2`, `K7_R1_2`
(rate 1/2), `K7_R1_3` (rate 1/3) and `K5_R1_5` (rate 1/5), in order of increasing
redundancy. Flip, insert, and delete are swept to 20%; erasures are far more
correctable, so that axis is swept to 80%. Each channel axis has two metrics
(edit distance, run length between edits) in two units (per info bit, per coded
bit).

### Flip channel (bit substitutions)

![edit distance per info bit vs flip rate](plots/edit_vs_flip_per_info_bit.png)

Edit distance per info bit. Every code holds at zero up to a threshold, then
climbs, and tolerance scales with redundancy: `K5_R1_5` barely reaches 0.02 even
at 20% flips, `K7_R1_3` turns up around 10%, and the two rate-1/2 codes turn up
near 6–8% (`K7_R1_2` degrades fastest past its knee, ending highest at ~0.28).

![edit distance per coded bit vs flip rate](plots/edit_vs_flip_per_coded_bit.png)

The same data per coded bit (each curve ÷ its rate `n`) — the fair cross-code
comparison, since the x-axis is per coded bit. The ranking is unchanged but the
gaps widen: `K5_R1_5` stays under 0.005 while `K7_R1_2` reaches ~0.14.

![mean info bits between edits vs flip rate](plots/runlen_vs_flip_per_info_bit.png)

Mean run length between edits, in info bits (linear, adaptive y-cap ~1140). Each
code's run length is effectively unbounded (off the top) below its threshold,
then drops off a cliff: `K3_R1_2` near 2.5%, `K7_R1_2` near 5%, `K7_R1_3` near
10%, `K5_R1_5` near 15%. This reads directly as "how long a clean run you can
expect" at a given flip rate.

![mean coded bits between edits vs flip rate](plots/runlen_vs_flip_per_coded_bit.png)

The same in coded bits (`n`× larger, cap ~3100); identical cliff positions. Use
this view when you care about bits on the wire rather than payload delivered.

### Insert channel (spurious bits)

![edit distance per info bit vs insert rate](plots/edit_vs_insert_per_info_bit.png)

Edit distance per info bit. Insertions are the harshest impairment and the
curves are jagged — once the decoder mis-tracks an insertion the cost is large
and high-variance. The rate-1/2 codes spike earliest (`K3_R1_2` jumps to ~0.23
by 7%); all four converge toward 0.2–0.27 by 20%, so redundancy buys little in
info-bit terms here.

![edit distance per coded bit vs insert rate](plots/edit_vs_insert_per_coded_bit.png)

Per coded bit the codes separate clearly: `K5_R1_5` stays smooth and low (~0.05
at 20%), `K7_R1_3` next, and the rate-1/2 codes are highest and noisiest — so
extra redundancy does help per coded bit, even against insertions.

![mean info bits between edits vs insert rate](plots/runlen_vs_insert_per_info_bit.png)

Run length in info bits (cap ~95) collapses almost immediately: every code falls
below ~95 info bits per edit by ~3% insert rate and settles at ~5–10 by 7%. The
codes are nearly indistinguishable in this unit.

![mean coded bits between edits vs insert rate](plots/runlen_vs_insert_per_coded_bit.png)

In coded bits (cap ~225) the redundancy ordering reappears in the tail:
`K5_R1_5` sustains ~20–50 coded bits per edit out to 20%, `K7_R1_3` ~15–35, and
the rate-1/2 codes ~8–10.

### Delete channel (dropped bits)

![edit distance per info bit vs delete rate](plots/edit_vs_delete_per_info_bit.png)

Edit distance per info bit. The rate-1/2 codes (overlapping) climb fastest and
plateau near 0.30 by ~12%; `K7_R1_3` and `K5_R1_5` rise more gradually
(`K5_R1_5` lowest, ~0.25 at 20%, with some mid-range noise). The knee is earlier
than for flips — a dropped coded bit also stresses re-anchoring.

![edit distance per coded bit vs delete rate](plots/edit_vs_delete_per_coded_bit.png)

Per coded bit, a clean redundancy ranking: the rate-1/2 codes plateau ~0.15,
`K7_R1_3` ~0.09, `K5_R1_5` ~0.05 at 20%.

![mean info bits between edits vs delete rate](plots/runlen_vs_delete_per_info_bit.png)

Run length in info bits (cap ~75) drops steeply near the origin to ~3–5 bits per
edit in the tail. A few noise spikes survive 30 trials (e.g. `K5_R1_5` near 6%),
inherent to the high-variance failure regime.

![mean coded bits between edits vs delete rate](plots/runlen_vs_delete_per_coded_bit.png)

In coded bits (cap ~200) the stronger codes hold longer runs in the tail
(`K5_R1_5` ~20–37, `K7_R1_3` ~12, rate-1/2 ~6–8 coded bits per edit), with the
same occasional spikes.

### Erase channel (bits marked lost)

Erasures carry no wrong information — the decoder knows which symbols are
unknown — so the codes tolerate far higher erasure rates than flips or indels,
each failing only as the erasure rate nears its `1 - rate` capacity limit. Note
the wider x-axis (to 80%).

![edit distance per info bit vs erase rate](plots/edit_vs_erase_per_info_bit.png)

Edit distance per info bit — clean S-curves with knees at the capacity limits.
The rate-1/2 codes (`K3_R1_2`, `K7_R1_2`) turn up around 45% (`K7_R1_2` a touch
earlier and steeper), `K7_R1_3` (rate 1/3) around 60%, and `K5_R1_5` (rate 1/5)
only past ~72%. Every code rides out 20% erasures at essentially zero edits.

![edit distance per coded bit vs erase rate](plots/edit_vs_erase_per_coded_bit.png)

Per coded bit, the same knees with curves scaled by rate: `K5_R1_5` stays under
0.03 even at 80% erasures, while the rate-1/2 codes reach ~0.18.

![mean info bits between edits vs erase rate](plots/runlen_vs_erase_per_info_bit.png)

Mean run length between edits, info bits (cap ~760). Each code holds an
effectively unbounded run until its capacity limit, then drops off a cliff; the
cliffs march rightward with redundancy (rate-1/2 near 45%, `K7_R1_3` near 60%,
`K5_R1_5` near 75%).

![mean coded bits between edits vs erase rate](plots/runlen_vs_erase_per_coded_bit.png)

The same in coded bits (cap ~1970); identical cliff positions, scaled by rate.
