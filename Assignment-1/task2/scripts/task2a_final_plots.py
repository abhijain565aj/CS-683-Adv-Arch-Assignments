#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_csv(path):
    df = pd.read_csv(path)
    required = {"stage", "size", "time_ms"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns: {sorted(missing)}")
    return df


def median_by_size(df):
    return (
        df.groupby("size", as_index=False)["time_ms"]
        .median()
        .sort_values("size")
    )


def select_naive(df):
    x = df[df["stage"].astype(str).str.lower() == "naive"].copy()
    if x.empty:
        raise ValueError("Could not find rows with stage='naive'")
    return median_by_size(x)


def select_best_sw_prefetch(df):
    x = df[df["stage"].astype(str).str.lower().str.contains("prefetch")].copy()
    if x.empty:
        raise ValueError("Could not find software-prefetch rows")

    idx = x.groupby("size")["time_ms"].idxmin()
    return x.loc[idx].sort_values("size").reset_index(drop=True)


def plot_single(data, label, title, output):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(data["size"], data["time_ms"], marker="o", linewidth=2, label=label)
    ax.set_xlabel("Matrix size (N)")
    ax.set_ylabel("Execution time (ms)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_comparison(series, title, output):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, data in series:
        ax.plot(data["size"], data["time_ms"], marker="o", linewidth=2, label=label)

    ax.set_xlabel("Matrix size (N)")
    ax.set_ylabel("Execution time (ms)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_speedup(naive_no_hw, sw_no_hw, output):
    merged = naive_no_hw[["size", "time_ms"]].merge(
        sw_no_hw[["size", "time_ms"]],
        on="size",
        suffixes=("_naive", "_sw"),
    )
    merged["speedup"] = merged["time_ms_naive"] / merged["time_ms_sw"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(merged["size"], merged["speedup"], marker="o", linewidth=2)
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Matrix size (N)")
    ax.set_ylabel("Speedup")
    ax.set_title("Software Prefetch Speedup with Hardware Prefetch Disabled")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)

    return merged


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot Task 2A results for naive+HW ON, naive+HW OFF, "
            "and software-prefetch+HW OFF."
        )
    )
    SCRIPT_DIR = Path(__file__).resolve().parent
    TASK2_ROOT = SCRIPT_DIR.parent
    parser.add_argument("--hw-on", required=True,
                        help="CSV collected with hardware prefetch enabled")
    parser.add_argument("--hw-off", required=True,
                        help="CSV collected with hardware prefetch disabled")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=TASK2_ROOT / "results" / "task2a_plots_final"
    )
    args = parser.parse_args()

    hw_on = load_csv(args.hw_on)
    hw_off = load_csv(args.hw_off)

    naive_hw_on = select_naive(hw_on)
    naive_hw_off = select_naive(hw_off)
    sw_hw_off = select_best_sw_prefetch(hw_off)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_single(
        naive_hw_on,
        "Naive, HW prefetch ON",
        "Naive SGEMM with Hardware Prefetch Enabled",
        outdir / "01_naive_hw_on.png",
    )

    plot_single(
        naive_hw_off,
        "Naive, HW prefetch OFF",
        "Naive SGEMM with All Prefetching Disabled",
        outdir / "02_naive_hw_off.png",
    )

    plot_single(
        sw_hw_off,
        "Software prefetch, HW prefetch OFF",
        "Software Prefetch with Hardware Prefetch Disabled",
        outdir / "03_sw_prefetch_hw_off.png",
    )

    plot_comparison(
        [
            ("Naive + HW prefetch ON", naive_hw_on),
            ("Software prefetch + HW prefetch OFF", sw_hw_off),
        ],
        "Hardware Prefetch vs Software Prefetch",
        outdir / "04_hw_vs_sw_prefetch.png",
    )

    plot_comparison(
        [
            ("Naive + HW prefetch ON", naive_hw_on),
            ("Naive + HW prefetch OFF", naive_hw_off),
            ("Software prefetch + HW prefetch OFF", sw_hw_off),
        ],
        "Task 2A Prefetch Comparison",
        outdir / "05_all_three_comparison.png",
    )

    speedup = plot_speedup(
        naive_hw_off,
        sw_hw_off,
        outdir / "06_sw_prefetch_speedup.png",
    )

    selected = naive_hw_on[["size", "time_ms"]].rename(
        columns={"time_ms": "naive_hw_on_ms"}
    )
    selected = selected.merge(
        naive_hw_off[["size", "time_ms"]].rename(
            columns={"time_ms": "naive_hw_off_ms"}
        ),
        on="size",
        how="outer",
    )
    selected = selected.merge(
        sw_hw_off[["size", "time_ms"]].rename(
            columns={"time_ms": "sw_prefetch_hw_off_ms"}
        ),
        on="size",
        how="outer",
    )
    selected = selected.merge(
        speedup[["size", "speedup"]],
        on="size",
        how="left",
    )
    selected.to_csv(outdir / "selected_results.csv", index=False)

    keep = [
        c for c in [
            "size",
            "distance_floats",
            "distance_cache_lines",
            "hint",
            "time_ms",
            "gflops",
        ]
        if c in sw_hw_off.columns
    ]
    sw_hw_off[keep].to_csv(outdir / "best_sw_prefetch_hw_off.csv", index=False)

    print(f"Plots written to: {outdir}")
    for p in sorted(outdir.glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
