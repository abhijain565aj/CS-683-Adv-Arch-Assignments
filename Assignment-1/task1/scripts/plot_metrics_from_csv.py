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

KEY_COLUMNS = {"H", "W", "K", "stage"}

METRIC_LABELS = {
    "speedup": "Speedup over naive",
    "time_ms": "Execution time (ms)",
    "gflops": "Performance (GFLOP/s)",
    "instructions": "Instructions",
    "cycles": "Cycles",
    "cpi": "CPI",
    "ipc": "IPC",
    "branches": "Branch instructions",
    "branches_per_output": "Branch instructions per output element",
    "branch_misses": "Branch misses",
    "branch_miss_rate": "Branch miss rate (%)",
    "cache_references": "Cache references",
    "cache_misses": "Cache misses",
    "l1_misses": "L1-D cache misses",
    "l1_mpki": "L1-D MPKI",
}

ALIASES = {
    "time": "time_ms",
    "execution time": "time_ms",
    "gflop/s": "gflops",
    "branch instructions": "branches",
    "branch_instruction": "branches",
    "branch_instructions": "branches",
    "branch-misses": "branch_misses",
    "branch miss rate": "branch_miss_rate",
    "branch miss percentage": "branch_miss_rate",
    "l1-d misses": "l1_misses",
    "l1-d mpki": "l1_mpki",
    "mpki": "l1_mpki",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Read convolution result CSV files and plot selected metrics "
            "against the naive implementation."
        )
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help=(
            "Metrics to plot, for example: --metrics branches l1_mpki. "
            "If omitted, the script asks interactively."
        ),
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        type=Path,
        help=(
            "Specific CSV files to read. By default, all *_results.csv files "
            "under the input directory are read."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "results" / "plots",
        help="Directory containing result CSV files (default: plots).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "plots" / "from_csv",
        help="Directory for generated plots (default: plots/from_csv).",
    )
    return parser.parse_args()


def find_csv_files(args):
    if args.csv:
        csv_files = [path.resolve() for path in args.csv]
    else:
        input_dir = args.input_dir.resolve()
        csv_files = sorted(input_dir.glob("*_results.csv"))

    missing = [path for path in csv_files if not path.is_file()]
    if missing:
        missing_names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"CSV file not found: {missing_names}")

    if not csv_files:
        raise FileNotFoundError(
            f"No *_results.csv files found in {args.input_dir.resolve()}."
        )

    return csv_files


def read_csv_files(csv_files):
    collected = defaultdict(lambda: defaultdict(list))

    for csv_path in csv_files:
        with csv_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)

            if reader.fieldnames is None:
                raise ValueError(f"{csv_path} has no header.")

            missing_keys = KEY_COLUMNS - set(reader.fieldnames)
            if missing_keys:
                missing = ", ".join(sorted(missing_keys))
                raise ValueError(f"{csv_path} is missing columns: {missing}")

            metric_columns = [
                column for column in reader.fieldnames if column not in KEY_COLUMNS
            ]

            for row_number, row in enumerate(reader, start=2):
                stage = row["stage"].strip().lower()
                if stage not in STAGES:
                    continue

                try:
                    height = int(row["H"])
                    width = int(row["W"])
                    kernel_size = int(row["K"])
                except ValueError as error:
                    raise ValueError(
                        f"Invalid H, W or K in {csv_path}, row {row_number}."
                    ) from error

                key = (height, width, kernel_size, stage)

                for metric in metric_columns:
                    text = row[metric].strip()
                    if not text or text.upper() == "N/A":
                        continue

                    try:
                        value = float(text)
                    except ValueError:
                        continue

                    if math.isfinite(value):
                        collected[key][metric].append(value)

    data = {}
    for key, metric_values in collected.items():
        data[key] = {
            metric: sum(values) / len(values)
            for metric, values in metric_values.items()
        }

    derive_speedup_from_time(data)
    return data


def derive_speedup_from_time(data):
    workloads = {
        (height, width, kernel_size)
        for height, width, kernel_size, _ in data
    }

    for height, width, kernel_size in workloads:
        naive_key = (height, width, kernel_size, "naive")
        naive_time = data.get(naive_key, {}).get("time_ms")

        if naive_time is None or naive_time == 0:
            continue

        for stage in STAGES:
            key = (height, width, kernel_size, stage)
            stage_time = data.get(key, {}).get("time_ms")

            if stage_time is not None and stage_time != 0:
                data[key].setdefault("speedup", naive_time / stage_time)


def available_metrics(data):
    metrics = set()
    for values in data.values():
        metrics.update(values)
    return sorted(metrics)


def normalize_metric_name(metric):
    normalized = metric.strip().lower()
    return ALIASES.get(normalized, normalized)


def choose_metrics(requested_metrics, available):
    if requested_metrics:
        selected = [normalize_metric_name(metric) for metric in requested_metrics]
    else:
        print("\nAvailable metrics:")
        for index, metric in enumerate(available, start=1):
            label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
            print(f"  {index:2d}. {metric:<22} {label}")

        answer = input(
            "\nEnter one or more metric names or numbers, separated by commas: "
        ).strip()

        selected = []
        for item in answer.split(","):
            item = item.strip()
            if item.isdigit():
                position = int(item)
                if position < 1 or position > len(available):
                    raise ValueError(f"Metric number {position} is invalid.")
                selected.append(available[position - 1])
            else:
                selected.append(normalize_metric_name(item))

    selected = list(dict.fromkeys(selected))

    unavailable = [metric for metric in selected if metric not in available]
    if unavailable:
        raise ValueError(
            "Unavailable metric(s): "
            + ", ".join(unavailable)
            + ". Available metrics: "
            + ", ".join(available)
        )

    if "speedup" in available and "speedup" not in selected:
        selected.insert(0, "speedup")

    return selected


def get_value(data, image_size, kernel_size, stage, metric):
    key = (image_size, image_size, kernel_size, stage)

    if key not in data or metric not in data[key]:
        raise ValueError(
            f"Missing {metric} for H=W={image_size}, "
            f"K={kernel_size}, stage={stage}."
        )

    return data[key][metric]


def metric_label(metric):
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def finish_plot(x_values, x_label, metric, title, all_y_values, output_path):
    plt.xlabel(x_label)
    plt.ylabel(metric_label(metric))
    plt.title(title)
    plt.xticks(x_values, [str(value) for value in x_values])

    finite_values = [
        abs(value)
        for value in all_y_values
        if math.isfinite(value)
    ]
    if finite_values and max(finite_values) >= 100000:
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_combined(data, metric, sweep, output_path):
    if sweep == "image":
        x_values = IMAGE_SIZES
        x_label = "Image size (H = W), K = 5"

        def value_for(stage, x_value):
            return get_value(data, x_value, FIXED_KERNEL_SIZE, stage, metric)

        title_suffix = "Image Size"
    else:
        x_values = KERNEL_SIZES
        x_label = "Kernel size, H = W = 2048"

        def value_for(stage, x_value):
            return get_value(data, FIXED_IMAGE_SIZE, x_value, stage, metric)

        title_suffix = "Kernel Size"

    plotted_values = []
    plotted_stages = OPTIMIZED_STAGES if metric == "speedup" else STAGES

    plt.figure(figsize=(9, 5.5))
    for stage in plotted_stages:
        values = [value_for(stage, x_value) for x_value in x_values]
        plotted_values.extend(values)
        plt.plot(x_values, values, marker="o", linewidth=2, label=stage)

    if metric == "speedup":
        plt.axhline(
            1.0,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="naive baseline (1×)",
        )
        plotted_values.append(1.0)

    finish_plot(
        x_values,
        x_label,
        metric,
        f"{metric_label(metric)} vs. {title_suffix}",
        plotted_values,
        output_path,
    )


def plot_stage_against_naive(data, metric, stage, sweep, output_path):
    if sweep == "image":
        x_values = IMAGE_SIZES
        x_label = "Image size (H = W), K = 5"

        def value_for(stage_name, x_value):
            return get_value(
                data, x_value, FIXED_KERNEL_SIZE, stage_name, metric
            )

        title_suffix = "Image Size"
    else:
        x_values = KERNEL_SIZES
        x_label = "Kernel size, H = W = 2048"

        def value_for(stage_name, x_value):
            return get_value(
                data, FIXED_IMAGE_SIZE, x_value, stage_name, metric
            )

        title_suffix = "Kernel Size"

    stage_values = [value_for(stage, x_value) for x_value in x_values]
    plotted_values = list(stage_values)

    plt.figure(figsize=(8, 5))
    plt.plot(
        x_values,
        stage_values,
        marker="o",
        linewidth=2,
        color="tab:blue",
        label=stage,
    )

    if metric == "speedup":
        plt.axhline(
            1.0,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="naive baseline (1×)",
        )
        plotted_values.append(1.0)
    else:
        naive_values = [value_for("naive", x_value) for x_value in x_values]
        plotted_values.extend(naive_values)
        plt.plot(
            x_values,
            naive_values,
            marker="o",
            linewidth=2,
            linestyle="--",
            color="black",
            label="naive",
        )

    finish_plot(
        x_values,
        x_label,
        metric,
        f"{stage} vs. Naive: {metric_label(metric)} vs. {title_suffix}",
        plotted_values,
        output_path,
    )


def create_metric_plots(data, metric, output_root):
    metric_directory = output_root / metric
    created = []

    combined_size = metric_directory / f"all_stages_{metric}_vs_image_size_k5.png"
    combined_kernel = (
        metric_directory / f"all_stages_{metric}_vs_kernel_size_2048.png"
    )

    plot_combined(data, metric, "image", combined_size)
    plot_combined(data, metric, "kernel", combined_kernel)
    created.extend([combined_size, combined_kernel])

    for stage in OPTIMIZED_STAGES:
        size_path = (
            metric_directory / f"{stage}_vs_naive_{metric}_image_size_k5.png"
        )
        kernel_path = (
            metric_directory / f"{stage}_vs_naive_{metric}_kernel_size_2048.png"
        )

        plot_stage_against_naive(data, metric, stage, "image", size_path)
        plot_stage_against_naive(data, metric, stage, "kernel", kernel_path)
        created.extend([size_path, kernel_path])

    return created


def main():
    args = parse_arguments()

    try:
        csv_files = find_csv_files(args)
        data = read_csv_files(csv_files)
        available = available_metrics(data)

        if not available:
            raise ValueError("No numeric metrics were found in the CSV files.")

        selected_metrics = choose_metrics(args.metrics, available)
        output_root = args.output_dir.resolve()

        print("\nReading:")
        for csv_file in csv_files:
            print(f"  {csv_file}")

        print("\nPlotting:")
        for metric in selected_metrics:
            print(f"  {metric}")

        created_files = []
        for metric in selected_metrics:
            created_files.extend(
                create_metric_plots(data, metric, output_root)
            )

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        return 1

    print(f"\nCreated {len(created_files)} plots under:")
    print(f"  {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
