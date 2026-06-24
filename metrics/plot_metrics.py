#!/usr/bin/env python3
# MIT License - Copyright (c) 2026 Robyn Kirkman
"""Plot decoding-mistake metrics vs channel rate.

Reads the CSV produced by the dv_metrics harness and renders, for each channel
axis (flip / insert / delete), one curve per code. Three metrics are available:

  edit       - normalized edit (Levenshtein) distance: mistakes per bit.
  runlength  - mean run length between edits (1 / edit rate): the average number
               of bits that get through between mistakes - a sense of how long a
               transmission you can expect to push through cleanly.
  lock       - mean lock probability: the decoder's own confidence (0..1) that
               it is tracking a valid coded stream, averaged over the kept bits.

The edit and runlength metrics come in two units: per info bit (payload
delivered) or per coded bit (bits on the wire). The channel rate (x-axis) is per
coded bit; a rate-1/n code spends n coded bits per info bit, so the
per-coded-bit view compares codes fairly. Lock probability is unitless, so it is
rendered once per axis regardless of the --unit setting.

Run length is undefined where no edits were observed (a clean decode over the
whole measurement); those points are dropped and reported, since the true value
there is only a lower bound (> the bits measured), not infinity.

Usage:
    build/metrics/dv_metrics > metrics/metrics.csv
    python3 metrics/plot_metrics.py metrics/metrics.csv -o metrics/plots/
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


def _opt_float(row, col):
    """Parse an optional float column, or None if absent (older CSVs predating
    the metric)."""
    return float(row[col]) if row.get(col) else None


def load(path):
    # series[axis][code] -> {"edit": [(rate, edit_rate)...],
    #                        "lock": [(rate, mean_lock)...],
    #                        "detect": [(rate, mean_detect)...], "n": code_n}
    # Each metric carries its own (rate, value) points, since the harness sweeps
    # a separate rate grid per metric (the "metric" column tags each row). Older
    # CSVs without that column carry every metric on one grid, so a blank metric
    # is treated as contributing to all three (guarded by the per-column value
    # being present).
    def entry():
        return {"edit": [], "lock": [], "detect": [], "n": None}

    series = defaultdict(lambda: defaultdict(entry))
    columns = {"edit": "edit_rate", "lock": "mean_lock", "detect": "mean_detect"}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            e = series[row["axis"]][row["code"]]
            e["n"] = code_n(row)
            tag = (row.get("metric") or "").strip()
            rate = float(row["rate"])
            for metric, column in columns.items():
                if tag and tag != metric:
                    continue
                value = _opt_float(row, column)
                if value is not None:
                    e[metric].append((rate, value))
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


def collect_ys(by_code, metric, per_coded):
    """Every defined y-value across all codes for one (metric, unit), used to size
    a shared y-axis against a peer CSV."""
    ys = []
    for entry in by_code.values():
        for _r, ir in entry["edit"]:
            y = y_value(metric, per_coded, ir, entry["n"])
            if y is not None:
                ys.append(y)
    return ys


def plot_axis(axis, by_code, outdir, metric, unit, logy, ymax, peer_ys=None):
    per_coded = unit == "coded"
    ylabel = LABELS[(metric, per_coded)]
    fig, ax = plt.subplots(figsize=(7, 5))
    dropped = 0
    all_ys = []
    for code, entry in by_code.items():
        pts = []
        for r, ir in sorted(entry["edit"]):
            y = y_value(metric, per_coded, ir, entry["n"])
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
    # percentile, adapting to the info/coded scale; --ymax overrides. With a peer
    # CSV (--match) the top is widened to whichever side has the larger range, so
    # the paired untuned/tuned plots share a vertical scale for direct comparison.
    cap = None
    matched = False
    if not logy:
        ax.set_ylim(bottom=0)
        if ymax is not None:
            cap = ymax
        elif metric == "runlength":
            # Both sides cap at their own 75th percentile; the pair takes the
            # larger so neither curve is clipped relative to the other.
            caps = [c for c in (percentile(all_ys, 75),
                                percentile(peer_ys, 75) if peer_ys else None)
                    if c is not None]
            cap = max(caps) if caps else None
            matched = peer_ys is not None and cap is not None
        elif peer_ys is not None:
            # edit: autoscaled per CSV by default; pin the top to the larger of
            # the two data ranges (+5% margin, matching matplotlib's autoscale).
            combined = all_ys + peer_ys
            cap = max(combined) * 1.05 if combined else None
            matched = cap is not None
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
        notes.append(f"y {'matched to peer' if matched else 'capped'} at "
                     f"{cap:.4g}")
    note = f"  ({'; '.join(notes)})" if notes else ""
    print(f"wrote {out}{note}")


# Unitless probability metrics, each a single curve-per-code plot per axis. The
# key also names the per-metric point list in a series entry (entry[metric]).
PROB_METRICS = {
    "lock": {"ylabel": "mean P(decoder locked)",
             "title": "Lock probability", "missing": "mean_lock"},
    "detect": {"ylabel": "mean P(code detected)",
               "title": "Detection probability", "missing": "mean_detect"},
}


def plot_prob_axis(axis, by_code, outdir, metric):
    """Render a unitless probability metric (lock or detect) vs channel rate, one
    curve per code. Both live in [0, 1], so there is a single plot per axis."""
    spec = PROB_METRICS[metric]
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for code, entry in by_code.items():
        pts = sorted(entry[metric])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                    label=code)
            plotted = True
    if not plotted:
        plt.close(fig)
        print(f"skipped {metric}_vs_{axis}: no {spec['missing']} column in CSV")
        return
    ax.set_xlabel(f"channel {axis} rate per coded bit")
    ax.set_ylabel(spec["ylabel"])
    ax.set_title(f"{spec['title']} vs {axis} rate")
    ax.set_ylim(-0.02, 1.02)
    if metric == "lock":
        ax.set_xlim(left=0.0, right=1.0)  # show the full rate range to capacity
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(outdir, f"{metric}_vs_{axis}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="CSV from dv_metrics")
    ap.add_argument("-o", "--outdir", default=".", help="output directory")
    ap.add_argument("--metric",
                    choices=["edit", "runlength", "lock", "detect", "all"],
                    default="all", help="edit distance, mean run length between "
                    "edits, mean lock probability, mean detect probability, or "
                    "all (default)")
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
    ap.add_argument("--match", metavar="PEER.csv", default=None,
                    help="match each edit/runlength plot's y-axis to whichever of "
                    "this CSV or the peer has the larger range, so paired "
                    "untuned/tuned plots share a vertical scale (ignored by the "
                    "unitless lock/detect plots and overridden by --ymax)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    series = load(args.csv)
    if not series:
        raise SystemExit("no rows found in CSV")
    peer_series = load(args.match) if args.match else None
    metrics = (["edit", "runlength", "lock", "detect"] if args.metric == "all"
               else [args.metric])
    units = ["info", "coded"] if args.unit == "both" else [args.unit]
    for axis, by_code in series.items():
        peer_by_code = peer_series.get(axis) if peer_series else None
        for metric in metrics:
            if metric in PROB_METRICS:
                plot_prob_axis(axis, by_code, args.outdir, metric)
                continue
            for unit in units:
                peer_ys = (collect_ys(peer_by_code, metric, unit == "coded")
                           if peer_by_code else None)
                plot_axis(axis, by_code, args.outdir, metric, unit, args.logy,
                          args.ymax, peer_ys)


if __name__ == "__main__":
    main()
