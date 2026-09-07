#!/usr/bin/env python3
"""
Generate Task 2A plots and a text summary from one or more collector CSVs.

Example:
  python3 task2a_plot.py task2a_results_default.csv --outdir task2a_plots

With hardware-prefetcher ON/OFF datasets:
  python3 task2a_plot.py task2a_results_default.csv task2a_results_hw_off.csv \
      --outdir task2a_plots
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import math
import statistics
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is required: python3 -m pip install matplotlib", file=sys.stderr)
    raise


NUMERIC_FIELDS = {
    "size": int,
    "distance_floats": float,
    "distance_cache_lines": float,
    "reps": int,
    "warmups": int,
    "time_ms": float,
    "gflops": float,
    "speedup": float,
    "instructions_total": float,
    "instructions_per_call": float,
    "l1d_misses_total": float,
    "l1d_misses_per_call": float,
    "l2_misses_total": float,
    "l2_misses_per_call": float,
    "llc_misses_total": float,
    "llc_misses_per_call": float,
    "sw_prefetch_access_total": float,
    "sw_prefetch_access_per_call": float,
    "checksum": float,
}


def load_rows(paths):
    rows = []
    for path in paths:
        with Path(path).open(newline="") as f:
            for row in csv.DictReader(f):
                for k, conv in NUMERIC_FIELDS.items():
                    if k in row and row[k] not in ("", None):
                        try:
                            row[k] = conv(row[k])
                        except ValueError:
                            pass
                rows.append(row)
    return rows


def gmean(vals):
    vals = [float(v) for v in vals if float(v) > 0]
    if not vals:
        return float("-inf")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def save_line_plot(xys, xlabel, ylabel, title, path, xscale=None):
    plt.figure()
    for label, xs, ys in xys:
        plt.plot(xs, ys, marker="o", label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if xscale:
        plt.xscale(xscale)
    if len(xys) > 1:
        plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def available(rows, field):
    return any(r.get(field, "") not in ("", None) for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    SCRIPT_DIR = Path(__file__).resolve().parent
    TASK2_ROOT = SCRIPT_DIR.parent
    ap.add_argument(
        "--outdir",
        type=Path,
        default=TASK2_ROOT / "results" / "task2a_plots"
    )
    args = ap.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit("No rows found")

    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)

    hw_labels = sorted(set(str(r.get("hw_prefetch", "default")) for r in rows))
    default_label = "default" if "default" in hw_labels else hw_labels[0]

    # 1) Best software-prefetch speedup vs matrix size.
    best_by_hw_size = defaultdict(list)
    for r in rows:
        if r.get("stage") == "prefetch":
            best_by_hw_size[(r["hw_prefetch"], r["size"])].append(r)

    xys = []
    for hw in hw_labels:
        sizes = sorted({size for (label, size) in best_by_hw_size if label == hw})
        xs, ys = [], []
        for size in sizes:
            candidates = best_by_hw_size[(hw, size)]
            best = max(candidates, key=lambda r: float(r["speedup"]))
            xs.append(size)
            ys.append(float(best["speedup"]))
        if xs:
            xys.append((f"best SW prefetch ({hw})", xs, ys))
    if xys:
        save_line_plot(
            xys,
            "Matrix size N (M=N=K)",
            "Speedup over no-prefetch baseline",
            "Task 2A: Best Software-Prefetch Speedup vs Matrix Size",
            out / "task2a_speedup_vs_size.png",
        )

    # Prefer explicit distance sweep; use full-grid T0 if necessary.
    distance_rows = [
        r for r in rows
        if r.get("stage") == "prefetch"
        and r.get("hw_prefetch") == default_label
        and r.get("sweep") == "distance"
    ]
    if not distance_rows:
        full = [
            r for r in rows
            if r.get("stage") == "prefetch"
            and r.get("hw_prefetch") == default_label
            and r.get("sweep") == "full_grid"
        ]
        if full:
            preferred_hint = "T0" if any(r.get("hint") == "T0" for r in full) else full[0]["hint"]
            distance_rows = [r for r in full if r.get("hint") == preferred_hint]

    # 2) Speedup vs prefetch distance.
    if distance_rows:
        xys = []
        for size in sorted({r["size"] for r in distance_rows}):
            rs = sorted(
                [r for r in distance_rows if r["size"] == size],
                key=lambda r: r["distance_floats"],
            )
            xys.append((
                f"N={size}",
                [r["distance_floats"] for r in rs],
                [r["speedup"] for r in rs],
            ))
        save_line_plot(
            xys,
            "Prefetch distance (floats)",
            "Speedup over no-prefetch baseline",
            "Task 2A: Speedup vs Prefetch Distance",
            out / "task2a_prefetch_distance.png",
            xscale="log",
        )

    # Determine a fair fixed distance for level comparison.
    best_distance = None
    if distance_rows:
        grouped = defaultdict(list)
        for r in distance_rows:
            grouped[r["distance_floats"]].append(float(r["speedup"]))
        best_distance = max(grouped, key=lambda d: gmean(grouped[d]))

    level_rows = [
        r for r in rows
        if r.get("stage") == "prefetch"
        and r.get("hw_prefetch") == default_label
        and r.get("sweep") == "level"
    ]
    if not level_rows and best_distance is not None:
        level_rows = [
            r for r in rows
            if r.get("stage") == "prefetch"
            and r.get("hw_prefetch") == default_label
            and r.get("sweep") == "full_grid"
            and float(r["distance_floats"]) == float(best_distance)
        ]

    # 3) Speedup vs cache fill/locality level.
    if level_rows:
        hints = ["T0", "T1", "T2", "NTA"]
        sizes = sorted({r["size"] for r in level_rows})
        plt.figure()
        width = 0.18
        x = list(range(len(sizes)))
        for idx, hint in enumerate(hints):
            ys = []
            for size in sizes:
                cand = [
                    r for r in level_rows
                    if r["size"] == size and r.get("hint") == hint
                ]
                ys.append(float(cand[0]["speedup"]) if cand else float("nan"))
            offsets = [v + (idx - 1.5) * width for v in x]
            plt.bar(offsets, ys, width=width, label=hint)
        plt.xticks(x, [str(s) for s in sizes])
        plt.xlabel("Matrix size N (M=N=K)")
        plt.ylabel("Speedup over no-prefetch baseline")
        plt.title("Task 2A: Speedup vs Prefetch Cache Fill Level")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "task2a_prefetch_level.png", dpi=180)
        plt.close()

    # 4-7) CPU metrics vs distance.
    metric_specs = [
        ("l1d_misses_per_call", "L1D misses per kernel call", "task2a_l1d_misses_vs_distance.png"),
        ("l2_misses_per_call", "L2 misses per kernel call", "task2a_l2_misses_vs_distance.png"),
        ("llc_misses_per_call", "LLC misses per kernel call", "task2a_llc_misses_vs_distance.png"),
        ("sw_prefetch_access_per_call", "SW prefetch accesses per kernel call", "task2a_sw_prefetch_access_vs_distance.png"),
    ]
    for field, ylabel, filename in metric_specs:
        if not distance_rows or not available(distance_rows, field):
            continue
        xys = []
        for size in sorted({r["size"] for r in distance_rows}):
            rs = sorted(
                [
                    r for r in distance_rows
                    if r["size"] == size and r.get(field, "") not in ("", None)
                ],
                key=lambda r: r["distance_floats"],
            )
            if rs:
                xys.append((
                    f"N={size}",
                    [r["distance_floats"] for r in rs],
                    [r[field] for r in rs],
                ))
        if xys:
            save_line_plot(
                xys,
                "Prefetch distance (floats)",
                ylabel,
                f"Task 2A: {ylabel} vs Prefetch Distance",
                out / filename,
                xscale="log",
            )

    # 8) Hardware prefetcher comparison if multiple labels are supplied.
    if len(hw_labels) >= 2:
        xys = []
        for hw in hw_labels:
            sizes = sorted({
                r["size"] for r in rows
                if r.get("stage") == "prefetch" and r.get("hw_prefetch") == hw
            })
            xs, ys = [], []
            for size in sizes:
                cand = [
                    r for r in rows
                    if r.get("stage") == "prefetch"
                    and r.get("hw_prefetch") == hw
                    and r["size"] == size
                ]
                if cand:
                    best = max(cand, key=lambda r: float(r["speedup"]))
                    xs.append(size)
                    ys.append(float(best["speedup"]))
            if xs:
                xys.append((hw, xs, ys))
        if len(xys) >= 2:
            save_line_plot(
                xys,
                "Matrix size N (M=N=K)",
                "Best SW-prefetch speedup",
                "Task 2A: Hardware Prefetcher ON/Default vs OFF",
                out / "task2a_hw_prefetch_on_off.png",
            )

    # Text summary.
    summary = []
    summary.append(f"Input CSVs: {', '.join(args.csv)}")
    summary.append(f"Hardware-prefetch labels: {', '.join(hw_labels)}")
    summary.append("")

    if best_distance is not None:
        summary.append(
            f"Best prefetch distance in the distance sweep "
            f"(geometric mean across sizes): {best_distance:g} floats "
            f"= {best_distance / 16.0:g} cache lines"
        )

    if level_rows:
        by_hint = defaultdict(list)
        for r in level_rows:
            by_hint[r["hint"]].append(float(r["speedup"]))
        best_hint = max(by_hint, key=lambda h: gmean(by_hint[h]))
        summary.append(
            f"Best prefetch level in the level sweep "
            f"(geometric mean across sizes): {best_hint}"
        )

    summary.append("")
    summary.append("Best collected software-prefetch configuration per matrix size:")
    for hw in hw_labels:
        summary.append(f"  [{hw}]")
        sizes = sorted({
            r["size"] for r in rows
            if r.get("stage") == "prefetch" and r.get("hw_prefetch") == hw
        })
        for size in sizes:
            cand = [
                r for r in rows
                if r.get("stage") == "prefetch"
                and r.get("hw_prefetch") == hw
                and r["size"] == size
            ]
            best = max(cand, key=lambda r: float(r["speedup"]))
            summary.append(
                f"    N={size}: {best['speedup']:.4f}x, "
                f"distance={best['distance_floats']:g} floats, hint={best['hint']}, "
                f"time={best['time_ms']:.4f} ms"
            )

    summary_path = out / "task2a_summary.txt"
    summary_path.write_text("\n".join(summary) + "\n")

    print(f"Wrote plots and summary to: {out}")
    for p in sorted(out.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
