#!/usr/bin/env python3
# MIT License - Copyright (c) 2026 Robyn Kirkman
"""Plot decoding-mistake metrics vs channel rate.

Reads the CSV produced by the dv_metrics harness and renders, for each channel
axis (flip / insert / delete), one curve per code. Two metrics are available:

  edit       - normalized edit (Levenshtein) distance: mistakes per bit.
  runlength  - mean run length between edits (1 / edit rate): the average number
               of bits that get through between mistakes - a sense of how long a
               transmission you can expect to push through cleanly.

and two units: per info bit (payload delivered) or per coded bit (bits on the
wire). The channel rate (x-axis) is per coded bit; a rate-1/n code spends n
coded bits per info bit, so the per-coded-bit view compares codes fairly.

Run length is undefined where no edits were observed (a clean decode over the
whole measurement); those points are dropped and reported, since the true value
there is only a lower bound (> the bits measured), not infinity.

Usage:
    build/bench/dv_metrics > metrics.csv
    python3 bench/plot_metrics.py metrics.csv -o plots/
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed
import matplotlib.pyplot as plt


def percentile(values, p):
    """Linear-interpolated p-th percentile of values (no numpy dependency)."""
    s = sorted(values)
    if not s:
        return None
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def code_n(row):
    """Coded bits per info bit for a row: the explicit code_n column if the
    harness emitted it, else parsed from the code name (..._R1_<n>)."""
    if row.get("code_n"):
        return int(row["code_n"])
    return int(row["code"].rsplit("_", 1)[1])


def load(path):
    # series[axis][code] -> (rates, info_rates, n), preserving encounter order.
    series = defaultdict(lambda: defaultdict(lambda: ([], [], [None])))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rates, info_rates, n = series[row["axis"]][row["code"]]
            rates.append(float(row["rate"]))
            info_rates.append(float(row["edit_rate"]))
            n[0] = code_n(row)
    return series


# (metric, per_coded) -> y-axis label
LABELS = {
    ("edit", False): "edit distance per info bit",
    ("edit", True): "edit distance per coded bit",
    ("runlength", False): "mean info bits between edits",
    ("runlength", True): "mean coded bits between edits",
}


def y_value(metric, per_coded, info_rate, n):
    """Map a per-info-bit edit rate to the requested metric/unit, or None if the
    point is undefined (run length with no observed edits)."""
    if metric == "edit":
        return info_rate / n if per_coded else info_rate
    if info_rate == 0.0:
        return None  # run length is unbounded - only a lower bound is known
    return (n if per_coded else 1) / info_rate


def plot_axis(axis, by_code, outdir, metric, unit, logy, ymax):
    per_coded = unit == "coded"
    ylabel = LABELS[(metric, per_coded)]
    fig, ax = plt.subplots(figsize=(7, 5))
    dropped = 0
    all_ys = []
    for code, (rates, info_rates, n) in by_code.items():
        pts = []
        for r, ir in sorted(zip(rates, info_rates)):
            y = y_value(metric, per_coded, ir, n[0])
            if y is None:
                dropped += 1
            else:
                pts.append((r, y))
                all_ys.append(y)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                    label=code)
    ax.set_xlabel(f"channel {axis} rate per coded bit")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Decoding mistakes vs {axis} rate")
    if logy:
        ax.set_yscale("symlog", linthresh=1e-5) if metric == "edit" \
            else ax.set_yscale("log")

    # Cap the y-axis so a few near-vertical low-rate spikes don't flatten the
    # readable range. Default (run length only): clip linearly at the 75th
    # percentile, adapting to the info/coded scale; --ymax overrides.
    cap = None
    if not logy:
        ax.set_ylim(bottom=0)
        if ymax is not None:
            cap = ymax
        elif metric == "runlength":
            cap = percentile(all_ys, 75)
        if cap:
            ax.set_ylim(top=cap)

    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    name = "runlen" if metric == "runlength" else "edit"
    out = os.path.join(outdir, f"{name}_vs_{axis}_per_{unit}_bit.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    notes = []
    if dropped:
        notes.append(f"{dropped} zero-edit points dropped")
    if cap:
        notes.append(f"y capped at {cap:.0f}")
    note = f"  ({'; '.join(notes)})" if notes else ""
    print(f"wrote {out}{note}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="CSV from dv_metrics")
    ap.add_argument("-o", "--outdir", default=".", help="output directory")
    ap.add_argument("--metric", choices=["edit", "runlength", "both"],
                    default="both", help="edit distance, mean run length between "
                    "edits, or both (default)")
    ap.add_argument("--unit", choices=["info", "coded", "both"], default="both",
                    help="normalize per info bit, per coded bit, or both "
                    "(default)")
    ap.add_argument("--logy", action="store_true",
                    help="log y-axis (recommended for run length, which spans "
                    "several orders of magnitude); default linear")
    ap.add_argument("--ymax", type=float, default=None,
                    help="cap the linear y-axis at this value (default: run "
                    "length auto-caps at the 75th percentile so low-rate spikes "
                    "don't flatten the plot)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    series = load(args.csv)
    if not series:
        raise SystemExit("no rows found in CSV")
    metrics = ["edit", "runlength"] if args.metric == "both" else [args.metric]
    units = ["info", "coded"] if args.unit == "both" else [args.unit]
    for axis, by_code in series.items():
        for metric in metrics:
            for unit in units:
                plot_axis(axis, by_code, args.outdir, metric, unit, args.logy,
                          args.ymax)


if __name__ == "__main__":
    main()
