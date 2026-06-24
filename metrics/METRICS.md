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

The harness runs in three variations (see [Two variations](#two-variations) and
[Detection](#detection) below). Two measure decoding performance — `matched`,
where the decoder anticipates the channel, and `pegged`, where it does not — and
the third, `detect`, measures blind detection, which uses no decoder model. They
write to `metrics/tuned/`, `metrics/untuned/`, and `metrics/detect/`.

```sh
# Build the harness (off by default) and run each variation to its own CSV.
cmake -S . -B build -DDRIFT_VITERBI_BUILD_BENCH=ON
cmake --build build --target dv_metrics
# dv_metrics <trials> <info_bits> <seed> <variation=pegged|matched|detect>
#   (defaults: 50 1000 0xC0FFEE pegged)
build/metrics/dv_metrics 30 4000 0xC0FFEE matched > metrics/tuned/metrics.csv
build/metrics/dv_metrics 30 4000 0xC0FFEE pegged  > metrics/untuned/metrics.csv
build/metrics/dv_metrics 30 4000 0xC0FFEE detect  > metrics/detect/metrics.csv

# Plot the metrics each CSV carries (one curve per code). Needs matplotlib:
python3 -m venv .venv && .venv/bin/pip install matplotlib
# --match shares each edit/runlen plot's y-axis across the pair (the larger
# range wins) so the side-by-side untuned/tuned plots compare by eye directly.
.venv/bin/python metrics/plot_metrics.py metrics/tuned/metrics.csv   -o metrics/tuned/plots/   --match metrics/untuned/metrics.csv
.venv/bin/python metrics/plot_metrics.py metrics/untuned/metrics.csv -o metrics/untuned/plots/ --match metrics/tuned/metrics.csv
.venv/bin/python metrics/plot_metrics.py metrics/detect/metrics.csv  -o metrics/detect/plots/ --metric detect
```

All three share a `seed`, so a given (code, axis, rate) point sees the same
channel realizations across variations. The two decoding variations sample
different rate grids, though — each tuned to its own differently-shaped curves —
so they line up only where their grids share a rate; the seed still makes every
run reproducible.

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
`tuned/plots/{edit,runlen}_vs_{flip,insert,delete,erase}_per_{info,coded}_bit.png` —
and the two probabilities are unitless, one plot per axis —
`tuned/plots/{lock,detect}_vs_{flip,insert,delete,erase}.png`:

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
just one (`--unit` is ignored by the unitless lock/detect plots). Pass
`--match OTHER.csv` to size each edit/runlen plot's y-axis to whichever of the
two CSVs has the larger range, so the side-by-side untuned/tuned plots below
share a vertical scale and compare by eye (the lock/detect plots already do,
fixed to `[0, 1]`).

## Two variations

The decoding metrics — edit distance, run length, lock — come in two variations
that differ in the decoder's channel model: what the decoder believes about the
impairment it is facing. The channel, codes, and seed are the same; the
`dv_stream_params` probabilities the decoder is built with change, and because
that reshapes the curves, each variation samples its **own rate grid** tuned to
where its curves actually move.

- **untuned** (`metrics/untuned/`, the `pegged` model) — every impairment
  probability is pegged at a flat 1%, regardless of the rate actually being
  swept, with drift tracking always on. The decoder is never told what the
  channel is doing, so as a rate climbs it meets something increasingly
  unexpected. This shows how the decoder handles **channels it cannot
  anticipate**.
- **tuned** (`metrics/tuned/`, the `matched` model) — the active impairment's
  probability is set to the swept rate, and drift is tracked only when that
  impairment is an insertion or deletion. The decoder's model matches the
  channel at every point. This shows how the decoder handles **channels it can
  anticipate** — its best case.

The matched decoder always does at least as well; the gap is widest where a
pegged probability is most wrong (high flip/indel rates, where the matched edit
knees fall much later and matched lock barely dips) and narrowest for erasures,
which carry no wrong information for either model to misjudge. Blind detection is
neither — it uses no decoder model, lives in `metrics/detect/`, and is described
under [Detection](#detection).

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
actually moves, so the x-axes differ between plots: for the untuned set, edit and
run length run to ~20% (to ~90% on erase, which is far more correctable) and lock
runs the full range to 100%. The tuned set uses its own grids — edit knees fall
later, so it runs higher (~32%), and lock barely dips.

Each decoding metric below is shown as a table with the **untuned** (pegged-model)
figure on the left and its **tuned** (matched-model) counterpart on the right —
the same metric with the decoder unable to anticipate the channel (left) versus
anticipating it (right). The captions describe the untuned (left) case, the
harder one; see [Two variations](#two-variations) for how the tuned column
differs. Detection has its own section after these.

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

| Untuned — unanticipated channel | Tuned — anticipated channel |
|---|---|
| <img src="untuned/plots/edit_vs_flip_per_info_bit.png" alt="edit/info vs flip, untuned" width="420"> | <img src="tuned/plots/edit_vs_flip_per_info_bit.png" alt="edit/info vs flip, tuned" width="420"> |
| <img src="untuned/plots/edit_vs_insert_per_info_bit.png" alt="edit/info vs insert, untuned" width="420"> | <img src="tuned/plots/edit_vs_insert_per_info_bit.png" alt="edit/info vs insert, tuned" width="420"> |
| <img src="untuned/plots/edit_vs_delete_per_info_bit.png" alt="edit/info vs delete, untuned" width="420"> | <img src="tuned/plots/edit_vs_delete_per_info_bit.png" alt="edit/info vs delete, tuned" width="420"> |
| <img src="untuned/plots/edit_vs_erase_per_info_bit.png" alt="edit/info vs erase, untuned" width="420"> | <img src="tuned/plots/edit_vs_erase_per_info_bit.png" alt="edit/info vs erase, tuned" width="420"> |

Per coded bit (bits on the wire, each curve ÷ its rate `n`):

| Untuned — unanticipated channel | Tuned — anticipated channel |
|---|---|
| <img src="untuned/plots/edit_vs_flip_per_coded_bit.png" alt="edit/coded vs flip, untuned" width="420"> | <img src="tuned/plots/edit_vs_flip_per_coded_bit.png" alt="edit/coded vs flip, tuned" width="420"> |
| <img src="untuned/plots/edit_vs_insert_per_coded_bit.png" alt="edit/coded vs insert, untuned" width="420"> | <img src="tuned/plots/edit_vs_insert_per_coded_bit.png" alt="edit/coded vs insert, tuned" width="420"> |
| <img src="untuned/plots/edit_vs_delete_per_coded_bit.png" alt="edit/coded vs delete, untuned" width="420"> | <img src="tuned/plots/edit_vs_delete_per_coded_bit.png" alt="edit/coded vs delete, tuned" width="420"> |
| <img src="untuned/plots/edit_vs_erase_per_coded_bit.png" alt="edit/coded vs erase, untuned" width="420"> | <img src="tuned/plots/edit_vs_erase_per_coded_bit.png" alt="edit/coded vs erase, tuned" width="420"> |

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

| Untuned — unanticipated channel | Tuned — anticipated channel |
|---|---|
| <img src="untuned/plots/runlen_vs_flip_per_info_bit.png" alt="runlen/info vs flip, untuned" width="420"> | <img src="tuned/plots/runlen_vs_flip_per_info_bit.png" alt="runlen/info vs flip, tuned" width="420"> |
| <img src="untuned/plots/runlen_vs_insert_per_info_bit.png" alt="runlen/info vs insert, untuned" width="420"> | <img src="tuned/plots/runlen_vs_insert_per_info_bit.png" alt="runlen/info vs insert, tuned" width="420"> |
| <img src="untuned/plots/runlen_vs_delete_per_info_bit.png" alt="runlen/info vs delete, untuned" width="420"> | <img src="tuned/plots/runlen_vs_delete_per_info_bit.png" alt="runlen/info vs delete, tuned" width="420"> |
| <img src="untuned/plots/runlen_vs_erase_per_info_bit.png" alt="runlen/info vs erase, untuned" width="420"> | <img src="tuned/plots/runlen_vs_erase_per_info_bit.png" alt="runlen/info vs erase, tuned" width="420"> |

Per coded bit (`n`× larger; identical cliff positions):

| Untuned — unanticipated channel | Tuned — anticipated channel |
|---|---|
| <img src="untuned/plots/runlen_vs_flip_per_coded_bit.png" alt="runlen/coded vs flip, untuned" width="420"> | <img src="tuned/plots/runlen_vs_flip_per_coded_bit.png" alt="runlen/coded vs flip, tuned" width="420"> |
| <img src="untuned/plots/runlen_vs_insert_per_coded_bit.png" alt="runlen/coded vs insert, untuned" width="420"> | <img src="tuned/plots/runlen_vs_insert_per_coded_bit.png" alt="runlen/coded vs insert, tuned" width="420"> |
| <img src="untuned/plots/runlen_vs_delete_per_coded_bit.png" alt="runlen/coded vs delete, untuned" width="420"> | <img src="tuned/plots/runlen_vs_delete_per_coded_bit.png" alt="runlen/coded vs delete, tuned" width="420"> |
| <img src="untuned/plots/runlen_vs_erase_per_coded_bit.png" alt="runlen/coded vs erase, untuned" width="420"> | <img src="tuned/plots/runlen_vs_erase_per_coded_bit.png" alt="runlen/coded vs erase, tuned" width="420"> |

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

| Untuned — unanticipated channel | Tuned — anticipated channel |
|---|---|
| <img src="untuned/plots/lock_vs_flip.png" alt="lock vs flip, untuned" width="420"> | <img src="tuned/plots/lock_vs_flip.png" alt="lock vs flip, tuned" width="420"> |
| <img src="untuned/plots/lock_vs_insert.png" alt="lock vs insert, untuned" width="420"> | <img src="tuned/plots/lock_vs_insert.png" alt="lock vs insert, tuned" width="420"> |
| <img src="untuned/plots/lock_vs_delete.png" alt="lock vs delete, untuned" width="420"> | <img src="tuned/plots/lock_vs_delete.png" alt="lock vs delete, tuned" width="420"> |
| <img src="untuned/plots/lock_vs_erase.png" alt="lock vs erase, untuned" width="420"> | <img src="tuned/plots/lock_vs_erase.png" alt="lock vs erase, tuned" width="420"> |

## Detection

Detection stands apart from the two decoding variations: it uses no decoder
model, so it is computed once and lives in `metrics/detect/`, shared by both.

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

<img src="detect/plots/detect_vs_flip.png" alt="detection probability vs flip rate" width="600">
<img src="detect/plots/detect_vs_insert.png" alt="detection probability vs insert rate" width="600">
<img src="detect/plots/detect_vs_delete.png" alt="detection probability vs delete rate" width="600">
<img src="detect/plots/detect_vs_erase.png" alt="detection probability vs erase rate" width="600">
