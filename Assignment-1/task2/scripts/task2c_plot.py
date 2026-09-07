#!/usr/bin/env python3
"""Generate Task 2C plots and an optional final 2A-vs-2B-vs-2C comparison."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is required: python3 -m pip install matplotlib", file=sys.stderr)
    raise

NUMERIC_FIELDS = {
    "width_bits",
    "size",
    "distance_floats",
    "distance_cache_lines",
    "time_ms",
    "gflops",
    "speedup_vs_naive",
    "speedup_vs_no_sw_prefetch",
    "speedup_vs_simd_only_256",
    "instructions_per_kernel",
    "cycles_per_kernel",
    "ipc",
    "l1d_misses_per_kernel",
    "l2_misses_per_kernel",
    "llc_misses_per_kernel",
    "sw_prefetch_access_per_kernel",
}


def to_float(v):
    try:
        if v is None or str(v).strip() == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def load_csv(path):
    rows = []
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            row["_source"] = str(path)
            for field in NUMERIC_FIELDS:
                if field in row:
                    row[field] = to_float(row[field])
            rows.append(row)
    return rows


def uniq_best(rows, key_fields, value_field):
    best = {}
    for r in rows:
        value = to_float(r.get(value_field))
        if value is None:
            continue
        key = tuple(r.get(k) for k in key_fields)
        if key not in best or value > to_float(best[key].get(value_field)):
            best[key] = r
    return list(best.values())


def save_line(series, xlabel, ylabel, title, path, logy=False):
    if not series:
        return False
    plt.figure()
    for label, xs, ys in series:
        if not xs:
            continue
        plt.plot(xs, ys, marker="o", label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if logy:
        plt.yscale("log")
    if len(series) > 1:
        plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return True


def save_bar(labels, values, ylabel, title, path):
    if not labels:
        return False
    plt.figure()
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return True


def gmean(values):
    vals = [float(v) for v in values if v is not None and float(v) > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(x) for x in vals) / len(vals))


def choose_hw_label(rows):
    labels = sorted({r.get("hw_prefetch", "") for r in rows if r.get("hw_prefetch", "")})
    if "on" in labels:
        return "on"
    return labels[0] if labels else "default"


def rows_for_hw(rows, label):
    return [r for r in rows if r.get("hw_prefetch") == label]


def best_combined_by_size(rows):
    rows = [
        r for r in rows
        if r.get("stage") == "combined" and to_float(r.get("speedup_vs_naive")) is not None
    ]
    return uniq_best(rows, ["size"], "speedup_vs_naive")


def final_source_best(paths, stage_name):
    """Return {size: best speedup} for Task2A/2B CSVs with tolerant field naming."""
    result = {}
    for path in paths:
        try:
            rows = load_csv(path)
        except FileNotFoundError:
            continue
        for r in rows:
            size = to_float(r.get("size"))
            if size is None:
                size = to_float(r.get("N"))
            if size is None:
                continue

            speed = to_float(r.get("speedup_vs_naive"))
            if speed is None:
                speed = to_float(r.get("speedup"))
            if speed is None:
                continue

            # Ignore obvious baseline rows where possible.
            stage = str(r.get("stage", "")).lower()
            if stage_name == "prefetch" and stage == "naive":
                continue
            if stage_name == "simd" and stage in ("naive", "no simd"):
                continue

            n = int(size)
            result[n] = max(result.get(n, float("-inf")), speed)
    return result


def main():
    ap = argparse.ArgumentParser()
    SCRIPT_DIR = Path(__file__).resolve().parent
    TASK2_ROOT = SCRIPT_DIR.parent
    ap.add_argument(
        "--outdir",
        type=Path,
        default=TASK2_ROOT / "results" / "task2c_plots"
    )
    ap.add_argument("csv", nargs="+", help="one or more Task 2C CSV files, e.g. HW on/off")
    ap.add_argument("--focus-size", type=int, default=None)
    ap.add_argument("--task2a", action="append", default=[], help="Task 2A CSV; may be repeated")
    ap.add_argument("--task2b", action="append", default=[], help="Task 2B CSV; may be repeated")
    args = ap.parse_args()

    rows = []
    for p in args.csv:
        rows.extend(load_csv(p))
    if not rows:
        raise SystemExit("No Task 2C rows found")

    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)

    sizes = sorted({int(r["size"]) for r in rows if to_float(r.get("size")) is not None})
    focus_size = args.focus_size if args.focus_size is not None else max(sizes)
    primary_hw = choose_hw_label(rows)
    primary = rows_for_hw(rows, primary_hw)

    generated = []

    # 1. Best combined speedup vs size, one line per HW-prefetch state.
    series = []
    for hw in sorted({r.get("hw_prefetch", "") for r in rows}):
        subset = rows_for_hw(rows, hw)
        best = sorted(best_combined_by_size(subset), key=lambda r: r["size"])
        if best:
            series.append((
                f"HW prefetch {hw}",
                [int(r["size"]) for r in best],
                [float(r["speedup_vs_naive"]) for r in best],
            ))
    p = out / "task2c_best_speedup_vs_size.png"
    if save_line(series, "Matrix size N (M=N=K)", "Best speedup over naive",
                 "Task 2C: Best Combined Speedup vs Matrix Size", p):
        generated.append(p)

    # 2. Prefetch-distance effect for 256-bit combined kernel.
    distance_rows = [
        r for r in primary
        if r.get("stage") == "combined"
        and int(r.get("width_bits") or 0) == 256
        and r.get("sweep") == "distance"
        and to_float(r.get("speedup_vs_no_sw_prefetch")) is not None
    ]
    if not distance_rows:
        # Full-grid fallback: best hint for each size+distance.
        candidates = [
            r for r in primary
            if r.get("stage") == "combined"
            and int(r.get("width_bits") or 0) == 256
            and r.get("sweep") == "full_grid"
        ]
        distance_rows = uniq_best(candidates, ["size", "distance_floats"], "speedup_vs_no_sw_prefetch")

    series = []
    for size in sizes:
        rs = sorted(
            [r for r in distance_rows if int(r["size"]) == size],
            key=lambda r: r["distance_floats"],
        )
        if rs:
            series.append((
                f"N={size}",
                [int(r["distance_floats"]) for r in rs],
                [float(r["speedup_vs_no_sw_prefetch"]) for r in rs],
            ))
    p = out / "task2c_prefetch_distance.png"
    if save_line(series, "Prefetch distance (floats)", "Speedup vs same kernel without SW prefetch",
                 f"Task 2C: Prefetch Distance ({primary_hw} HW Prefetch)", p):
        generated.append(p)

    # 3. Prefetch-level effect. Use geometric-mean speedup across sizes for a clean summary bar.
    level_rows = [
        r for r in primary
        if r.get("stage") == "combined"
        and int(r.get("width_bits") or 0) == 256
        and r.get("sweep") == "level"
        and to_float(r.get("speedup_vs_no_sw_prefetch")) is not None
    ]
    if not level_rows:
        candidates = [
            r for r in primary
            if r.get("stage") == "combined"
            and int(r.get("width_bits") or 0) == 256
            and r.get("sweep") == "full_grid"
        ]
        # Best distance for each size+hint.
        level_rows = uniq_best(candidates, ["size", "hint"], "speedup_vs_no_sw_prefetch")

    labels, values = [], []
    for hint in ["T0", "T1", "T2", "NTA"]:
        vals = [
            to_float(r.get("speedup_vs_no_sw_prefetch"))
            for r in level_rows if r.get("hint") == hint
        ]
        gm = gmean(vals)
        if gm is not None:
            labels.append(hint)
            values.append(gm)
    p = out / "task2c_prefetch_level.png"
    if save_bar(labels, values, "Geometric-mean speedup vs no SW prefetch",
                f"Task 2C: Prefetch Cache-Fill Hint ({primary_hw} HW Prefetch)", p):
        generated.append(p)

    # 4. SIMD width effect at chosen/best prefetch configuration.
    width_rows = [
        r for r in primary
        if r.get("stage") == "combined"
        and r.get("sweep") == "width"
        and to_float(r.get("speedup_vs_naive")) is not None
    ]
    if not width_rows:
        candidates = [
            r for r in primary
            if r.get("stage") == "combined" and r.get("sweep") == "full_grid"
        ]
        width_rows = uniq_best(candidates, ["size", "width_bits"], "speedup_vs_naive")

    series = []
    for size in sizes:
        rs = sorted(
            [r for r in width_rows if int(r["size"]) == size],
            key=lambda r: r["width_bits"],
        )
        if rs:
            series.append((
                f"N={size}",
                [int(r["width_bits"]) for r in rs],
                [float(r["speedup_vs_naive"]) for r in rs],
            ))
    p = out / "task2c_simd_width.png"
    if save_line(series, "SIMD width (bits)", "Speedup over naive",
                 f"Task 2C: SIMD Width with Software Prefetch ({primary_hw} HW Prefetch)", p):
        generated.append(p)

    # 5. Instruction count vs SIMD width at focus size.
    rs = sorted(
        [
            r for r in width_rows
            if int(r["size"]) == focus_size
            and to_float(r.get("instructions_per_kernel")) is not None
        ],
        key=lambda r: r["width_bits"],
    )
    if rs:
        p = out / "task2c_instructions_vs_width.png"
        if save_line(
            [(f"N={focus_size}", [int(r["width_bits"]) for r in rs],
              [float(r["instructions_per_kernel"]) for r in rs])],
            "SIMD width (bits)", "Instructions per matrix multiplication",
            f"Task 2C: Instruction Count vs SIMD Width (N={focus_size})", p,
        ):
            generated.append(p)

    # 6. Cache metrics vs prefetch distance for focus size, 256-bit.
    metric_specs = [
        ("l1d_misses_per_kernel", "L1D misses per matmul", "task2c_l1d_misses_vs_distance.png"),
        ("l2_misses_per_kernel", "L2 misses per matmul", "task2c_l2_misses_vs_distance.png"),
        ("llc_misses_per_kernel", "LLC misses per matmul", "task2c_llc_misses_vs_distance.png"),
        ("sw_prefetch_access_per_kernel", "Software-prefetch accesses per matmul", "task2c_sw_prefetch_access_vs_distance.png"),
    ]
    focus_distance_rows = [r for r in distance_rows if int(r["size"]) == focus_size]
    for field, ylabel, filename in metric_specs:
        rs = sorted(
            [r for r in focus_distance_rows if to_float(r.get(field)) is not None],
            key=lambda r: r["distance_floats"],
        )
        if not rs:
            continue
        p = out / filename
        if save_line(
            [(f"N={focus_size}", [int(r["distance_floats"]) for r in rs],
              [float(r[field]) for r in rs])],
            "Prefetch distance (floats)", ylabel,
            f"Task 2C: {ylabel} vs Prefetch Distance", p,
        ):
            generated.append(p)

    # 7. Direct HW-prefetch ON/OFF comparison at best combined configuration.
    if len({r.get("hw_prefetch", "") for r in rows}) >= 2:
        series = []
        for hw in sorted({r.get("hw_prefetch", "") for r in rows}):
            best = sorted(best_combined_by_size(rows_for_hw(rows, hw)), key=lambda r: r["size"])
            if best:
                series.append((
                    f"HW prefetch {hw}",
                    [int(r["size"]) for r in best],
                    [float(r["speedup_vs_naive"]) for r in best],
                ))
        p = out / "task2c_hw_prefetch_on_off.png"
        if save_line(series, "Matrix size N (M=N=K)", "Best speedup over naive",
                     "Task 2C: Hardware Prefetch ON vs OFF", p):
            generated.append(p)

    # 8. Required final summary: SW Prefetch vs SIMD vs SW Prefetch+SIMD.
    if args.task2a and args.task2b:
        a_best = final_source_best(args.task2a, "prefetch")
        b_best = final_source_best(args.task2b, "simd")
        c_best_rows = best_combined_by_size(primary)
        c_best = {int(r["size"]): float(r["speedup_vs_naive"]) for r in c_best_rows}

        common = sorted(set(a_best) & set(b_best) & set(c_best))
        if common:
            series = [
                ("Software prefetch", common, [a_best[n] for n in common]),
                ("SIMD", common, [b_best[n] for n in common]),
                ("Software prefetch + SIMD", common, [c_best[n] for n in common]),
            ]
            p = out / "task2_final_best_comparison.png"
            if save_line(series, "Matrix size N (M=N=K)", "Best speedup over naive",
                         "Task 2: Best Performance by Optimization Technique", p):
                generated.append(p)

    # Summary text.
    summary = []
    summary.append("Task 2C summary")
    summary.append(f"Primary HW-prefetch dataset used for parameter plots: {primary_hw}")
    summary.append(f"Focus size for counter plots: N={focus_size}")
    summary.append("")

    for hw in sorted({r.get("hw_prefetch", "") for r in rows}):
        summary.append(f"Hardware prefetch: {hw}")
        subset = rows_for_hw(rows, hw)
        best = sorted(best_combined_by_size(subset), key=lambda r: r["size"])
        for r in best:
            summary.append(
                f"  N={int(r['size'])}: best={float(r['speedup_vs_naive']):.4f}x, "
                f"width={int(r['width_bits'])}, distance={int(r['distance_floats'])}, "
                f"hint={r['hint']}, time={float(r['time_ms']):.4f} ms"
            )
        summary.append("")

    if level_rows:
        scored = []
        for hint in ["T0", "T1", "T2", "NTA"]:
            gm = gmean([
                to_float(r.get("speedup_vs_no_sw_prefetch"))
                for r in level_rows if r.get("hint") == hint
            ])
            if gm is not None:
                scored.append((gm, hint))
        if scored:
            gm, hint = max(scored)
            summary.append(f"Best prefetch hint by geometric mean: {hint} ({gm:.4f}x vs no-SW)")

    if distance_rows:
        by_d = defaultdict(list)
        for r in distance_rows:
            v = to_float(r.get("speedup_vs_no_sw_prefetch"))
            if v is not None:
                by_d[int(r["distance_floats"])].append(v)
        scores = [(gmean(v), d) for d, v in by_d.items() if gmean(v) is not None]
        if scores:
            gm, d = max(scores)
            summary.append(f"Best prefetch distance by geometric mean: {d} floats ({gm:.4f}x vs no-SW)")

    (out / "task2c_summary.txt").write_text("\n".join(summary) + "\n")
    generated.append(out / "task2c_summary.txt")

    print(f"Wrote Task 2C outputs to: {out}")
    for p in generated:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
