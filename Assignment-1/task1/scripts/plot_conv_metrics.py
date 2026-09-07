#!/usr/bin/env python3

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CONV_BIN = ROOT / "bin" / "conv"
PERF_SCRIPT = ROOT / "scripts" / "perf_compare.sh"
OUTPUT_DIR = ROOT / "plots"/ "results"

STAGES = ["naive", "reorder", "unroll", "tile", "simd", "optimized"]
OPTIMIZED_STAGES = ["reorder", "unroll", "tile", "simd", "optimized"]

IMAGE_SIZES = [512, 1024, 2048, 4096, 8192]
KERNEL_SIZES = [3, 5, 7, 9]
FIXED_KERNEL_SIZE = 5
FIXED_IMAGE_SIZE = 2048
PERF_KERNEL_CALLS = 25

BENCHMARK_METRICS = {"speedup", "time_ms", "gflops"}
PERF_METRICS = {
    "instructions",
    "cycles",
    "cpi",
    "ipc",
    "branches",
    "branches_per_output",
    "branch_misses",
    "branch_miss_rate",
    "cache_references",
    "cache_misses",
    "l1_misses",
    "l1_mpki",
}

METRIC_LABELS = {
    "speedup": "Speedup over naive",
    "time_ms": "Execution time (ms)",
    "gflops": "Performance (GFLOP/s)",
    "instructions": "Retired instructions",
    "cycles": "CPU cycles",
    "cpi": "Cycles per instruction (CPI)",
    "ipc": "Instructions per cycle (IPC)",
    "branches": "Retired branch instructions",
    "branches_per_output": "Approx. branch instructions per output element",
    "branch_misses": "Branch misses",
    "branch_miss_rate": "Branch miss rate (%)",
    "cache_references": "Cache references",
    "cache_misses": "Cache misses",
    "l1_misses": "L1-D load misses",
    "l1_mpki": "L1-D misses per kilo-instruction (MPKI)",
}

ALIASES = {
    "time": "time_ms",
    "execution_time": "time_ms",
    "gflop/s": "gflops",
    "gflops": "gflops",
    "branch_instructions": "branches",
    "branch instructions": "branches",
    "branch-misses": "branch_misses",
    "branch_miss_percentage": "branch_miss_rate",
    "branch miss percentage": "branch_miss_rate",
    "l1-d_mpki": "l1_mpki",
    "l1-d mpki": "l1_mpki",
    "mpki": "l1_mpki",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark all convolution stages and plot speedup plus one selected metric."
        )
    )
    parser.add_argument(
        "--metric",
        help="Metric to plot. If omitted, the script asks interactively.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--perf-repeats",
        type=int,
        default=3,
        help="Number of complete perf measurements for hardware metrics (default: 3).",
    )
    parser.add_argument(
        "--core-mode",
        choices=["p", "e", "both"],
        default="p",
        help="Core type used for perf measurements (default: p).",
    )
    parser.add_argument(
        "--cpu-pin",
        type=int,
        help="Optional logical CPU override for p/e core mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory in which plots and CSV files are written.",
    )
    return parser.parse_args()


def choose_metric(requested_metric):
    all_metrics = sorted(BENCHMARK_METRICS | PERF_METRICS)

    if requested_metric is None:
        print("Available metrics:")
        for index, metric in enumerate(all_metrics, start=1):
            print(f"  {index:2d}. {metric:<22} {METRIC_LABELS[metric]}")

        answer = input("\nEnter a metric name or number: ").strip()
        if answer.isdigit():
            position = int(answer)
            if 1 <= position <= len(all_metrics):
                requested_metric = all_metrics[position - 1]
            else:
                raise ValueError("Metric number is outside the displayed range.")
        else:
            requested_metric = answer

    metric = requested_metric.strip().lower()
    metric = ALIASES.get(metric, metric)

    if metric not in BENCHMARK_METRICS | PERF_METRICS:
        valid = ", ".join(all_metrics)
        raise ValueError(f"Unknown metric '{requested_metric}'. Choose one of: {valid}")

    return metric


def run_command(command, environment=None):
    command_text = " ".join(str(item) for item in command)
    print(f"  Running: {command_text}", flush=True)

    completed = subprocess.run(
        [str(item) for item in command],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if completed.returncode != 0:
        print(completed.stdout)
        raise RuntimeError(f"Command failed with exit code {completed.returncode}")

    return completed.stdout


def build_binaries(require_perf):
    print("\nBuilding convolution benchmark...")
    run_command(["make"])

    if not CONV_BIN.exists():
        raise FileNotFoundError(f"{CONV_BIN} was not produced by make.")

    if require_perf:
        if not (ROOT / "perf_main.cpp").exists():
            raise FileNotFoundError("perf_main.cpp was not found in the task1 directory.")
        if not PERF_SCRIPT.exists():
            raise FileNotFoundError("perf_compare.sh was not found in the task1 directory.")

        print("\nRebuilding the profiling executable from the current source files...")
        compile_command = [
            os.environ.get("CXX", "g++"),
            "-std=c++17",
            "-O2",
            "-fno-tree-vectorize",
            "-mavx2",
            "-mfma",
            "-Iinclude",
            "-Wall",
            "perf_main.cpp",
            "src/conv_naive.cpp",
            "src/conv_reorder.cpp",
            "src/conv_unroll.cpp",
            "src/conv_tile.cpp",
            "src/conv_simd.cpp",
            "src/conv_optimized.cpp",
            "-o",
            "bin/perf_conv",
        ]
        run_command(compile_command)


def parse_benchmark_output(output):
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    stage_pattern = re.compile(
        rf"^\s*(naive \(ref\)|reorder|unroll|tile|simd|optimized)"
        rf"\s+(yes|no)\s+({number})\s+({number})\s+({number})x\s*$"
    )

    results = {}

    for line in output.splitlines():
        match = stage_pattern.match(line)
        if not match:
            continue

        stage_name = match.group(1)
        if stage_name == "naive (ref)":
            stage_name = "naive"

        if match.group(2) != "yes":
            raise RuntimeError(f"Stage '{stage_name}' failed the correctness check.")

        results[stage_name] = {
            "time_ms": float(match.group(3)),
            "gflops": float(match.group(4)),
            "speedup": float(match.group(5)),
        }

    missing = [stage for stage in STAGES if stage not in results]
    if missing:
        print(output)
        raise RuntimeError(
            "Could not parse benchmark results for: " + ", ".join(missing)
        )

    return results


def parse_number(value):
    if value == "N/A":
        return float("nan")
    return float(value)


def parse_perf_output(output, image_size):
    results = {}
    inside_final_table = False

    for line in output.splitlines():
        if "Final comparison" in line:
            inside_final_table = True
            continue

        if not inside_final_table:
            continue

        fields = line.split()
        if len(fields) != 10 or fields[0] not in STAGES:
            continue

        stage = fields[0]
        instructions = parse_number(fields[1])
        cycles = parse_number(fields[2])
        ipc = parse_number(fields[3])
        branches = parse_number(fields[4])

        if instructions == 0:
            cpi = float("nan")
        else:
            cpi = cycles / instructions

        results[stage] = {
            "instructions": instructions,
            "cycles": cycles,
            "cpi": cpi,
            "ipc": ipc,
            "branches": branches,
            "branches_per_output": branches
            / (PERF_KERNEL_CALLS * image_size * image_size),
            "branch_miss_rate": parse_number(fields[5]),
            "cache_references": parse_number(fields[6]),
            "cache_misses": parse_number(fields[7]),
            "l1_misses": parse_number(fields[8]),
            "l1_mpki": parse_number(fields[9]),
        }

    missing = [stage for stage in STAGES if stage not in results]
    if missing:
        print(output)
        raise RuntimeError("Could not parse perf results for: " + ", ".join(missing))

    for stage in STAGES:
        branch_rate = results[stage]["branch_miss_rate"]
        branches = results[stage]["branches"]
        results[stage]["branch_misses"] = branches * branch_rate / 100.0

    return results


def collect_benchmark(image_size, kernel_size, seed):
    output = run_command(
        [
            CONV_BIN,
            "all",
            image_size,
            image_size,
            kernel_size,
            seed,
        ]
    )
    return parse_benchmark_output(output)


def collect_perf(image_size, kernel_size, seed, perf_repeats, core_mode, cpu_pin):
    environment = os.environ.copy()
    environment["PERF_REPEATS"] = str(perf_repeats)
    environment["CORE_MODE"] = core_mode

    if cpu_pin is not None:
        environment["CPU_PIN"] = str(cpu_pin)
    else:
        environment.pop("CPU_PIN", None)

    output = run_command(
        [
            "bash",
            PERF_SCRIPT,
            *STAGES,
            image_size,
            image_size,
            kernel_size,
            seed,
        ],
        environment=environment,
    )
    return parse_perf_output(output, image_size)


def collect_sweep(
    x_values,
    varying,
    selected_metric,
    seed,
    perf_repeats,
    core_mode,
    cpu_pin,
):
    benchmark_rows = []
    metric_rows = []
    needs_perf = selected_metric in PERF_METRICS

    for position, value in enumerate(x_values, start=1):
        if varying == "image_size":
            image_size = value
            kernel_size = FIXED_KERNEL_SIZE
        else:
            image_size = FIXED_IMAGE_SIZE
            kernel_size = value

        print(
            f"\n[{position}/{len(x_values)}] H=W={image_size}, K={kernel_size}",
            flush=True,
        )

        benchmark_result = collect_benchmark(image_size, kernel_size, seed)
        benchmark_rows.append(
            {
                "x": value,
                "H": image_size,
                "W": image_size,
                "K": kernel_size,
                "results": benchmark_result,
            }
        )

        if needs_perf:
            metric_result = collect_perf(
                image_size,
                kernel_size,
                seed,
                perf_repeats,
                core_mode,
                cpu_pin,
            )
        else:
            metric_result = benchmark_result

        metric_rows.append(
            {
                "x": value,
                "H": image_size,
                "W": image_size,
                "K": kernel_size,
                "results": metric_result,
            }
        )

    return benchmark_rows, metric_rows


def write_csv(path, rows, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["H", "W", "K", "stage", *metrics]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            for stage in STAGES:
                output_row = {
                    "H": row["H"],
                    "W": row["W"],
                    "K": row["K"],
                    "stage": stage,
                }
                for metric in metrics:
                    output_row[metric] = row["results"][stage][metric]
                writer.writerow(output_row)


def plot_metric(path, rows, metric, x_label, title, include_naive):
    stages = STAGES if include_naive else OPTIMIZED_STAGES
    x_values = [row["x"] for row in rows]

    plt.figure(figsize=(9, 5.5))

    for stage in stages:
        y_values = [row["results"][stage][metric] for row in rows]
        plt.plot(x_values, y_values, marker="o", linewidth=2, label=stage)

    plt.xlabel(x_label)
    plt.ylabel(METRIC_LABELS[metric])
    plt.title(title)
    plt.xticks(x_values, [str(value) for value in x_values])
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    args = parse_arguments()

    try:
        selected_metric = choose_metric(args.metric)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    if args.perf_repeats < 1:
        print("Error: --perf-repeats must be at least 1.", file=sys.stderr)
        return 2

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSelected metric: {selected_metric}")
    print("Sweep 1: H=W in 512, 1024, 2048, 4096, 8192 with K=5")
    print("Sweep 2: K in 3, 5, 7, 9 with H=W=2048")

    if selected_metric in PERF_METRICS:
        print(
            f"Perf mode: CORE_MODE={args.core_mode}, "
            f"PERF_REPEATS={args.perf_repeats}"
        )
        print("The 8192 runs can take considerable time.")

    try:
        build_binaries(selected_metric in PERF_METRICS)

        size_benchmark, size_metric = collect_sweep(
            IMAGE_SIZES,
            "image_size",
            selected_metric,
            args.seed,
            args.perf_repeats,
            args.core_mode,
            args.cpu_pin,
        )

        kernel_benchmark, kernel_metric = collect_sweep(
            KERNEL_SIZES,
            "kernel_size",
            selected_metric,
            args.seed,
            args.perf_repeats,
            args.core_mode,
            args.cpu_pin,
        )
    except (FileNotFoundError, RuntimeError) as error:
        print(f"\nError: {error}", file=sys.stderr)
        return 1

    speedup_size_plot = output_dir / "speedup_vs_image_size_k5.png"
    speedup_kernel_plot = output_dir / "speedup_vs_kernel_size_2048.png"

    plot_metric(
        speedup_size_plot,
        size_benchmark,
        "speedup",
        "Image size (H = W), K = 5",
        "Convolution Speedup vs. Image Size",
        include_naive=False,
    )
    plot_metric(
        speedup_kernel_plot,
        kernel_benchmark,
        "speedup",
        "Kernel size, H = W = 2048",
        "Convolution Speedup vs. Kernel Size",
        include_naive=False,
    )

    write_csv(
        output_dir / "benchmark_results.csv",
        size_benchmark + kernel_benchmark,
        ["time_ms", "gflops", "speedup"],
    )

    created_files = [speedup_size_plot, speedup_kernel_plot]

    if selected_metric != "speedup":
        metric_size_plot = output_dir / f"{selected_metric}_vs_image_size_k5.png"
        metric_kernel_plot = (
            output_dir / f"{selected_metric}_vs_kernel_size_2048.png"
        )

        plot_metric(
            metric_size_plot,
            size_metric,
            selected_metric,
            "Image size (H = W), K = 5",
            f"{METRIC_LABELS[selected_metric]} vs. Image Size",
            include_naive=True,
        )
        plot_metric(
            metric_kernel_plot,
            kernel_metric,
            selected_metric,
            "Kernel size, H = W = 2048",
            f"{METRIC_LABELS[selected_metric]} vs. Kernel Size",
            include_naive=True,
        )

        write_csv(
            output_dir / f"{selected_metric}_results.csv",
            size_metric + kernel_metric,
            [selected_metric],
        )
        created_files.extend([metric_size_plot, metric_kernel_plot])

    print("\nFinished. Created:")
    for path in created_files:
        print(f"  {path}")
    print(f"  {output_dir / 'benchmark_results.csv'}")
    if selected_metric != "speedup":
        print(f"  {output_dir / f'{selected_metric}_results.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
