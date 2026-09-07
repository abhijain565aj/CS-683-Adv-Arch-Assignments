#!/usr/bin/env python3

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["naive", "reorder", "unroll", "tile", "simd", "optimized"]
OPTIMIZED_STAGES = ["reorder", "unroll", "tile", "simd", "optimized"]

IMAGE_SIZES = [512, 1024, 2048, 4096, 8192]
KERNEL_SIZES = [3, 5, 7, 9]
FIXED_KERNEL_SIZE = 5
FIXED_IMAGE_SIZE = 2048


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Read branch data from CSV and compare each stage with naive."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "results" / "plots" / "branches_results.csv",
        help="Input CSV file (default: plots/branches_results.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "plots" / "branch_instructions",
        help="Output directory for the plots.",
    )
    return parser.parse_args()


def read_branch_data(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    grouped_values = defaultdict(list)

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"H", "W", "K", "stage", "branches"}

        if reader.fieldnames is None:
            raise ValueError("The CSV file has no header.")

        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing CSV columns: {missing}")

        for row_number, row in enumerate(reader, start=2):
            stage = row["stage"].strip()
            if stage not in STAGES:
                continue

            try:
                height = int(row["H"])
                width = int(row["W"])
                kernel_size = int(row["K"])
                branches = float(row["branches"])
            except ValueError as error:
                raise ValueError(
                    f"Invalid numeric value in CSV row {row_number}."
                ) from error

            if not math.isfinite(branches):
                raise ValueError(
                    f"Branch instructions are unavailable in CSV row {row_number}."
                )

            key = (height, width, kernel_size, stage)
            grouped_values[key].append(branches)

    data = {}
    for key, values in grouped_values.items():
        data[key] = sum(values) / len(values)

    return data


def get_value(data, image_size, kernel_size, stage):
    key = (image_size, image_size, kernel_size, stage)
    if key not in data:
        raise ValueError(
            "Missing result for "
            f"H=W={image_size}, K={kernel_size}, stage={stage}."
        )
    return data[key]


def configure_plot(x_values, x_label, title):
    plt.xlabel(x_label)
    plt.ylabel("Branch instructions")
    plt.title(title)
    plt.xticks(x_values, [str(value) for value in x_values])
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()


def plot_combined_size_sweep(data, output_path):
    plt.figure(figsize=(9, 5.5))

    for stage in STAGES:
        values = [
            get_value(data, image_size, FIXED_KERNEL_SIZE, stage)
            for image_size in IMAGE_SIZES
        ]
        plt.plot(IMAGE_SIZES, values, marker="o", linewidth=2, label=stage)

    configure_plot(
        IMAGE_SIZES,
        "Image size (H = W), K = 5",
        "Branch Instructions vs. Image Size",
    )
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_combined_kernel_sweep(data, output_path):
    plt.figure(figsize=(9, 5.5))

    for stage in STAGES:
        values = [
            get_value(data, FIXED_IMAGE_SIZE, kernel_size, stage)
            for kernel_size in KERNEL_SIZES
        ]
        plt.plot(KERNEL_SIZES, values, marker="o", linewidth=2, label=stage)

    configure_plot(
        KERNEL_SIZES,
        "Kernel size, H = W = 2048",
        "Branch Instructions vs. Kernel Size",
    )
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_stage_size_comparison(data, stage, output_path):
    naive_values = [
        get_value(data, image_size, FIXED_KERNEL_SIZE, "naive")
        for image_size in IMAGE_SIZES
    ]
    stage_values = [
        get_value(data, image_size, FIXED_KERNEL_SIZE, stage)
        for image_size in IMAGE_SIZES
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(
        IMAGE_SIZES,
        naive_values,
        marker="o",
        linewidth=2,
        linestyle="--",
        color="black",
        label="naive",
    )
    plt.plot(
        IMAGE_SIZES,
        stage_values,
        marker="o",
        linewidth=2,
        color="tab:blue",
        label=stage,
    )

    configure_plot(
        IMAGE_SIZES,
        "Image size (H = W), K = 5",
        f"{stage} vs. Naive: Branch Instructions",
    )
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_stage_kernel_comparison(data, stage, output_path):
    naive_values = [
        get_value(data, FIXED_IMAGE_SIZE, kernel_size, "naive")
        for kernel_size in KERNEL_SIZES
    ]
    stage_values = [
        get_value(data, FIXED_IMAGE_SIZE, kernel_size, stage)
        for kernel_size in KERNEL_SIZES
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(
        KERNEL_SIZES,
        naive_values,
        marker="o",
        linewidth=2,
        linestyle="--",
        color="black",
        label="naive",
    )
    plt.plot(
        KERNEL_SIZES,
        stage_values,
        marker="o",
        linewidth=2,
        color="tab:blue",
        label=stage,
    )

    configure_plot(
        KERNEL_SIZES,
        "Kernel size, H = W = 2048",
        f"{stage} vs. Naive: Branch Instructions",
    )
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_arguments()
    csv_path = args.csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        data = read_branch_data(csv_path)

        combined_size_plot = output_dir / "all_stages_branches_vs_image_size_k5.png"
        combined_kernel_plot = (
            output_dir / "all_stages_branches_vs_kernel_size_2048.png"
        )

        plot_combined_size_sweep(data, combined_size_plot)
        plot_combined_kernel_sweep(data, combined_kernel_plot)

        created_files = [combined_size_plot, combined_kernel_plot]

        for stage in OPTIMIZED_STAGES:
            stage_size_plot = (
                output_dir / f"{stage}_vs_naive_branches_image_size_k5.png"
            )
            stage_kernel_plot = (
                output_dir / f"{stage}_vs_naive_branches_kernel_size_2048.png"
            )

            plot_stage_size_comparison(data, stage, stage_size_plot)
            plot_stage_kernel_comparison(data, stage, stage_kernel_plot)
            created_files.extend([stage_size_plot, stage_kernel_plot])

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        return 1

    print("Created branch-instruction plots:")
    for path in created_files:
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
