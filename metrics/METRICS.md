# drift_viterbi metrics

`metrics/dv_metrics.c` measures the decoding-mistake rate as a function of the
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
build/metrics/dv_metrics > metrics/metrics.csv  # defaults: 12 trials, 4000 info bits
# build/metrics/dv_metrics <trials> <info_bits> <seed>   # e.g. 3 1000 1 for a quick run

# Plot the mistake metric vs each channel rate (one curve per code). Needs matplotlib:
python3 -m venv .venv && .venv/bin/pip install matplotlib
.venv/bin/python metrics/plot_metrics.py metrics/metrics.csv -o metrics/plots/
```

The default sweep takes a few minutes (the drift-tracking axes dominate); pass
smaller `trials`/`info_bits` for a faster, coarser run. The sweep is fanned out
across cores with OpenMP when available, and each point owns a seeded PRNG
stream, so a given `seed` reproduces every row's values exactly regardless of
thread count. Rows are streamed to stdout as each point finishes (so the CSV
grows as it runs), which means their order follows completion, not the code /
axis / rate grid — sort the file if you want a stable order; the plotter is
order-independent.

By default the plotter writes, per axis, the four metrics below. The first two
come in two units each —
`plots/{edit,runlen}_vs_{flip,insert,delete,erase}_per_{info,coded}_bit.png` —
and the two probabilities are unitless, one plot per axis —
`plots/{lock,detect}_vs_{flip,insert,delete,erase}.png`:

- **edit** — edit distance per bit (mistakes per bit).
- **runlen** — mean run length between edits (`1 / edit_rate`): the average bits
  that get through between mistakes, i.e. how long a transmission you can expect
  to push through cleanly. Points where no edits were observed are dropped (the
  value there is only a lower bound, not infinity).
- **lock** — mean lock probability (0–1): the decoder's own running confidence
  that it is tracking a valid coded stream, averaged over the kept bits.
- **detect** — mean detect probability (0–1): `dv_detect`'s blind confidence that
  the received buffer still carries *any* rate-1/n, constraint-length-k code,
  knowing neither the generators nor the sent bits.

For edit and runlen, since the channel rate (x-axis) is per coded bit, the
per-coded-bit view divides by the code's rate `n` so codes of different rates
compare fairly. Run-length plots are linear with an adaptive y-cap (the low-rate
spikes run off the top); pass `--logy` for a log axis or `--ymax N` to set the
cap. Pass `--metric edit|runlength|lock|detect` or `--unit info|coded` to emit
just one (`--unit` is ignored by the unitless lock/detect plots).

## Generated plots

The figures come from a `dv_metrics` sweep (the published set uses 30 trials ×
4000 info bits, default seed). In every plot the x-axis is the channel impairment
rate per coded bit and the four curves are the standard codes — `K3_R1_2`,
`K7_R1_2` (rate 1/2), `K7_R1_3` (rate 1/3) and `K5_R1_5` (rate 1/5), in order of
increasing redundancy.

The plots are grouped **by metric**, and each metric shows all four channel
impairments — flip, insert, delete, erase — side by side, so you read one metric
across the channels rather than one channel across the metrics. The swept rate
range is tuned per metric and per impairment to the band where that curve
actually moves, so the x-axes differ between plots: edit and run length run to
~20% (to ~90% on erase, which is far more correctable); lock runs the full range
to 100%; detection collapses so early that its axes stop within a few percent
(plus a high-erasure tail for the rebound noted below).

One thing to keep in mind throughout: the decoder is held to a **fixed,
channel-agnostic model** — every impairment pegged at 1%, regardless of the
actual rate being swept. It is never told what the channel is doing. So as a rate
climbs the decoder meets something increasingly unexpected, which is the stress
these curves measure; a decoder tuned to each rate would look better than this
but would not reflect a real deployment that cannot know the channel in advance.

### Edit distance (decoding mistakes per bit)

Normalized edit distance between the decoded and sent bits — the core
correctness metric. Every code holds near zero up to a per-code knee, then
climbs; tolerance scales with redundancy, so `K5_R1_5` holds out the longest and
the rate-1/2 codes turn up first. On the flip axis the knee (first rate past 1%
edit distance) sits near 1% for `K3_R1_2`, ~1.3% for `K7_R1_2`, ~2% for `K7_R1_3`
and ~3.5% for `K5_R1_5`. Because the decoder's model is pegged rather than matched
to the channel, the codes degrade harder past their knees than a channel-aware
decoder would: by 25% flips every code is up around 0.29–0.32 edits per info bit.
The per-coded-bit view divides each curve by its rate `n` (the fair cross-code
comparison, since the x-axis is per coded bit) and widens the gaps without
changing the ranking — at 25% flips `K5_R1_5` sits at ~0.06 against the rate-1/2
codes' ~0.16. Erasures, carrying no wrong information, are tolerated to far higher
rates — knees near each code's `1 - rate` capacity limit (~20% for `K3_R1_2`,
rising to ~48% for `K5_R1_5`).

Per info bit (payload delivered):

![edit distance per info bit vs flip rate](plots/edit_vs_flip_per_info_bit.png)
![edit distance per info bit vs insert rate](plots/edit_vs_insert_per_info_bit.png)
![edit distance per info bit vs delete rate](plots/edit_vs_delete_per_info_bit.png)
![edit distance per info bit vs erase rate](plots/edit_vs_erase_per_info_bit.png)

Per coded bit (bits on the wire, each curve ÷ its rate `n`):

![edit distance per coded bit vs flip rate](plots/edit_vs_flip_per_coded_bit.png)
![edit distance per coded bit vs insert rate](plots/edit_vs_insert_per_coded_bit.png)
![edit distance per coded bit vs delete rate](plots/edit_vs_delete_per_coded_bit.png)
![edit distance per coded bit vs erase rate](plots/edit_vs_erase_per_coded_bit.png)

### Run length between edits

The reciprocal of the edit rate (`1 / edit_rate`): the average number of bits
that get through between mistakes — how long a transmission you can expect to
push through cleanly. It is effectively unbounded below each code's knee (those
points, where no edits were seen, are dropped — the value there is only a lower
bound) and drops off a cliff at the knee, with the cliffs marching rightward with
redundancy. Plotted linearly with an adaptive y-cap so the low-rate spikes don't
flatten the readable range. Same knee positions as edit distance, since it is
derived from it.

Per info bit:

![mean info bits between edits vs flip rate](plots/runlen_vs_flip_per_info_bit.png)
![mean info bits between edits vs insert rate](plots/runlen_vs_insert_per_info_bit.png)
![mean info bits between edits vs delete rate](plots/runlen_vs_delete_per_info_bit.png)
![mean info bits between edits vs erase rate](plots/runlen_vs_erase_per_info_bit.png)

Per coded bit (`n`× larger; identical cliff positions):

![mean coded bits between edits vs flip rate](plots/runlen_vs_flip_per_coded_bit.png)
![mean coded bits between edits vs insert rate](plots/runlen_vs_insert_per_coded_bit.png)
![mean coded bits between edits vs delete rate](plots/runlen_vs_delete_per_coded_bit.png)
![mean coded bits between edits vs erase rate](plots/runlen_vs_erase_per_coded_bit.png)

### Lock probability

The decoder's own running confidence that it is tracking a valid coded stream,
averaged over the kept bits. Because the model is pegged at 1% rather than
matched to the channel, lock genuinely tracks the decoder being surprised: it
falls from ~1.0 along a descent and settles onto a plateau as each impairment
ramps, reflecting real loss of confidence rather than false comfort. The plateau
sits by redundancy — on the flip/insert/delete axes around 0.69 for `K7_R1_2`,
0.61 for `K3_R1_2`, 0.52 for `K7_R1_3` and 0.33 for `K5_R1_5` (the strongest code
spreads its few information bits thinnest, so a surprise dents its confidence
most). The descent is where the structure lives (the plots sample it densely);
past it the plateau is flat. Two endpoints are special: under deletion the stream
empties out near 100% and lock collapses to zero, and under flips a rate near
100% is a *deterministic* inversion — structured again — so lock partly recovers.
Erasures are a *known* unknown, so lock holds higher and longer before sliding to
zero as the rate nears capacity. Lock is the decoder's confidence, not a
correctness measure.

![lock probability vs flip rate](plots/lock_vs_flip.png)
![lock probability vs insert rate](plots/lock_vs_insert.png)
![lock probability vs delete rate](plots/lock_vs_delete.png)
![lock probability vs erase rate](plots/lock_vs_erase.png)

### Detection probability

`dv_detect`'s blind confidence that the received buffer still carries *any*
rate-1/n, constraint-length-k code, knowing neither the generators nor the sent
bits. It is the opposite story from lock: it falls well before decoding with a
known code fails, so it is an early warning, not a correctness measure. How early
is ordered by each code's relation window `n*(k+1)` (the span of bits its parity
checks couple): the wider the window, the more of the impaired stream it spans,
so the more fragile blind recovery is — the reverse of the decoder's redundancy
ranking. The short-window codes (`K3_R1_2` most of all) therefore stay detectable
far longer, which is why the flip/insert/delete axes run out past 20% to follow
their descent while the wide-window codes have long since collapsed.

Erasures follow the same window ordering and largely collapse: the wide-window
codes (`K7_R1_3`, `K5_R1_5`) are gone below ~5%, `K7_R1_2` drops off by ~8%, and
`K3_R1_2` holds longest — near 1.0 out to ~9%, then fading to zero by ~45%. Near
total erasure a small **spurious rebound** appears, this time for the wide-window
rate-1/3 and 1/5 codes (`K7_R1_3` reaches ~0.30, `K5_R1_5` ~0.11 at 92% erasure):
with almost every symbol blanked the recovered parity-check space degenerates and
trivially self-satisfies, so read that tail as a detector artifact, not a stream
becoming detectable again.

![detection probability vs flip rate](plots/detect_vs_flip.png)
![detection probability vs insert rate](plots/detect_vs_insert.png)
![detection probability vs delete rate](plots/detect_vs_delete.png)
![detection probability vs erase rate](plots/detect_vs_erase.png)
