#!/usr/bin/env python3
"""Run ChampSim experiments and turn simulator output into report data/plots."""

from __future__ import annotations

import argparse
import csv
import json
import re
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "report"
DATA = REPORT / "data"
PLOTS = REPORT / "plots"
LOGS = REPORT / "logs"
DEFAULT_WARMUP = 25_000_000
DEFAULT_SIMULATION = 25_000_000
NUMBER = r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf)"
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_CYAN = "\033[36m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"


def terminal_color(color: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{color}{text}{ANSI_RESET}"


@dataclass(frozen=True)
class Experiment:
    name: str
    task: str
    prefetcher: str
    replacement: str
    description: str


EXPERIMENTS: Dict[str, Experiment] = {
    "baseline": Experiment("baseline", "task1b", "no", "lru", "No prefetch, LRU"),
    "ip_stride": Experiment("ip_stride", "task1b", "ip_stride", "lru", "IP-stride, LRU"),
    "spp": Experiment("spp", "task1b", "spp", "lru", "SPP, LRU"),
    "rrip": Experiment("rrip", "task2a", "no", "rrip", "No prefetch, RRIP"),
    "irrip": Experiment("irrip", "task2a", "no", "irrip", "Instruction-conscious RRIP"),
    "dynamic_irrip": Experiment(
        "dynamic_irrip", "task2b", "no", "dynamic_irrip", "Dynamic I-RRIP"
    ),
    "combined": Experiment(
        "combined", "task3", "spp", "dynamic_irrip", "SPP with dynamic I-RRIP"
    ),
}

TASK_EXPERIMENTS = {
    "task1b": ("baseline", "ip_stride", "spp"),
    "task2a": ("rrip", "irrip"),
    "task2b": ("rrip", "irrip", "dynamic_irrip"),
    "task3": ("baseline", "spp", "dynamic_irrip", "combined"),
}

METRIC_PATTERNS = {
    "ipc": re.compile(rf"(?:CPU|Finished CPU)\s+(\d+).*?cumulative IPC:\s*({NUMBER})"),
    "l1d_load_mpki": re.compile(rf"L1D\s+LOAD.*?\bMPKI:\s*({NUMBER})"),
    "l2c_load_mpki": re.compile(rf"L2C\s+LOAD.*?\bMPKI:\s*({NUMBER})"),
    "l2i_load_mpki": re.compile(rf"L2C\s+INSTRUCTION LOAD MPKI:\s*({NUMBER})"),
    "l2d_load_mpki": re.compile(rf"L2C\s+DATA LOAD MPKI:\s*({NUMBER})"),
    "prefetch_accuracy_pct": re.compile(
        r"L1D\s+USEFUL LOAD PREFETCHES:.*?PREFETCH ISSUED TO LOWER LEVEL:.*?"
        rf"ACCURACY:\s*({NUMBER})"
    ),
    "prefetch_useful": re.compile(r"L1D\s+USEFUL LOAD PREFETCHES:\s*(\d+)"),
    "prefetch_issued": re.compile(
        r"L1D\s+USEFUL LOAD PREFETCHES:.*?PREFETCH ISSUED TO LOWER LEVEL:\s*(\d+)"
    ),
    "l1d_load_misses": re.compile(r"L1D\s+LOAD\s+ACCESS:\s*\d+\s+HIT:\s*\d+\s+MISS:\s*(\d+)"),
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_output(text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for name, pattern in METRIC_PATTERNS.items():
        matches = pattern.findall(text)
        if not matches:
            continue
        value = matches[-1]
        if name == "ipc":
            # The final CPU line is the ROI result; retain CPU 0 for single-core runs.
            cpu_values = {int(cpu): float(ipc) for cpu, ipc in matches}
            metrics[name] = cpu_values[min(cpu_values)]
        else:
            metrics[name] = float(value)

    if "prefetch_accuracy_pct" not in metrics and metrics.get("prefetch_issued", 0):
        metrics["prefetch_accuracy_pct"] = (
            100.0 * metrics.get("prefetch_useful", 0.0) / metrics["prefetch_issued"]
        )
    return metrics


def traces_from_argument(value: str) -> List[Path]:
    if value == "all":
        traces = sorted((ROOT / "traces").glob("*"))
    else:
        traces = [ROOT / "traces" / item for item in value.split(",")]
    traces = [path for path in traces if path.is_file()]
    if not traces:
        raise RuntimeError(f"No trace files found for --traces {value!r}")
    return traces


def ensure_experiment_available(experiment: Experiment) -> None:
    prefetch_file = ROOT / "prefetcher" / f"{experiment.prefetcher}.l1d_pref"
    replacement_file = ROOT / "replacement" / f"{experiment.replacement}.l2c_repl"
    missing = [
        str(path.relative_to(ROOT))
        for path in (prefetch_file, replacement_file)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            f"Experiment {experiment.name!r} is not implemented yet; missing "
            + ", ".join(missing)
        )


def run_streaming(
    command: List[str],
    cwd: Path,
    environment: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
) -> Tuple[int, str]:
    """Run a command while displaying and optionally recording its output."""
    command_text = f"$ {shlex.join(command)}"
    print(
        terminal_color(ANSI_BOLD + ANSI_CYAN, command_text),
        flush=True,
    )
    output: List[str] = []
    log_handle = log_path.open("w") if log_path is not None else open(os.devnull, "w")
    with log_handle:
        for attempt in range(3):
            try:
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                break
            except OSError as exc:
                if exc.errno != 26 or attempt == 2:
                    raise
                print("Executable is still being replaced; retrying...", flush=True)
                time.sleep(1)
        assert process.stdout is not None
        for line in process.stdout:
            output.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    return return_code, "".join(output)


def build_experiment(experiment: Experiment) -> Path:
    ensure_experiment_available(experiment)
    binary_name = f"runner-{experiment.name}"
    command = [
        str(ROOT / "build_champsim.sh"),
        experiment.prefetcher,
        experiment.replacement,
        binary_name,
    ]
    print(
        terminal_color(
            ANSI_BOLD + ANSI_GREEN,
            f"\n=== Building {experiment.name}: {experiment.description} ===",
        ),
        flush=True,
    )
    return_code, output = run_streaming(command, ROOT)
    if return_code:
        raise RuntimeError(
            f"Build failed for {experiment.name}:\n{output}"
        )
    candidates = sorted(ROOT.glob(f"bin/*-{binary_name}"))
    if not candidates:
        raise RuntimeError(f"Build succeeded but no binary ending in -{binary_name} exists")
    return candidates[-1]


def run_one(
    experiment: Experiment,
    trace: Path,
    warmup: int,
    simulation: int,
    threshold: Optional[float],
) -> Dict[str, object]:
    binary = build_experiment(experiment)
    label = f"{experiment.name}-{trace.stem}"
    if threshold is not None:
        label += f"-threshold-{threshold:g}"
    log_path = LOGS / f"{label}.log"
    LOGS.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary),
        "-warmup_instructions",
        str(warmup),
        "-simulation_instructions",
        str(simulation),
        "-traces",
        str(trace),
    ]
    environment = os.environ.copy()
    if threshold is not None:
        # Dynamic policies can consume this stable runner contract with
        # std::getenv("CHAMPSIM_THRESHOLD").
        environment["CHAMPSIM_THRESHOLD"] = str(threshold)
    print(
        terminal_color(
            ANSI_BOLD + ANSI_YELLOW,
            f"\n=== Running {experiment.name} on {trace.name} "
            f"(warmup={warmup}, simulation={simulation}) ===",
        ),
        flush=True,
    )
    return_code, output = run_streaming(
        command, ROOT, environment=environment, log_path=log_path
    )
    if return_code:
        raise RuntimeError(f"Simulation failed; see {log_path}: {return_code}")
    metrics = parse_output(output)
    if "ipc" not in metrics:
        raise RuntimeError(f"Could not find IPC in simulator output; see {log_path}")
    return {
        "trace": trace.name,
        "configuration": experiment.name,
        "description": experiment.description,
        "warmup_instructions": warmup,
        "simulation_instructions": simulation,
        "log": str(log_path.relative_to(ROOT)),
        **({"threshold": threshold} if threshold is not None else {}),
        **metrics,
    }


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def add_derived_metrics(rows: List[Dict[str, object]]) -> None:
    baselines: Dict[str, Dict[str, object]] = {}
    for row in rows:
        if row["configuration"] in {"baseline", "rrip"}:
            baselines.setdefault(str(row["trace"]), row)
    for row in rows:
        baseline = baselines.get(str(row["trace"]))
        if not baseline:
            continue
        ipc = float(row["ipc"])
        row["speedup"] = ipc / float(baseline["ipc"]) if baseline["ipc"] else None
        baseline_misses = baseline.get("l1d_load_misses")
        current_misses = row.get("l1d_load_misses")
        if baseline_misses is not None and current_misses is not None and float(baseline_misses):
            row["l1d_coverage_pct"] = (
                100.0 * (float(baseline_misses) - float(current_misses))
                / float(baseline_misses)
            )


def run_task(
    task: str,
    traces: List[Path],
    warmup: int,
    simulation: int,
    thresholds: List[float],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    names = TASK_EXPERIMENTS[task]
    for name in names:
        experiment = EXPERIMENTS[name]
        if name == "dynamic_irrip" and thresholds:
            for threshold in thresholds:
                for trace in traces:
                    rows.append(run_one(experiment, trace, warmup, simulation, threshold))
        else:
            for trace in traces:
                rows.append(run_one(experiment, trace, warmup, simulation, None))
    add_derived_metrics(rows)
    return rows


def write_task_data(task: str, rows: List[Dict[str, object]]) -> None:
    if task == "task1b":
        fields = ["trace", "configuration", "ipc", "speedup", "l1d_load_mpki",
                  "prefetch_accuracy_pct", "l1d_coverage_pct"]
        path = DATA / "task1b_results.csv"
    elif task == "task2a":
        fields = ["trace", "configuration", "ipc", "speedup", "l2c_load_mpki",
                  "l2i_load_mpki", "l2d_load_mpki"]
        path = DATA / "task2a_results.csv"
    elif task == "task2b":
        fields = ["trace", "configuration", "ipc", "speedup", "l2c_load_mpki",
                  "l2i_load_mpki", "l2d_load_mpki"]
        path = DATA / "task2b_sensitivity.csv"
    else:
        fields = ["trace", "configuration", "ipc", "speedup", "prefetch_accuracy_pct",
                  "l1d_coverage_pct", "l1d_load_mpki", "l2c_load_mpki",
                  "l2i_load_mpki", "l2d_load_mpki"]
        path = DATA / "task3_results.csv"
    write_csv(path, rows, fields)
    (DATA / f"{task}_raw.json").write_text(json.dumps(rows, indent=2))
    print(f"Wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


def plot_task(task: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib; install it in the Docker image") from exc
    data_path = {
        "task1b": DATA / "task1b_results.csv",
        "task2a": DATA / "task2a_results.csv",
        "task2b": DATA / "task2b_sensitivity.csv",
        "task3": DATA / "task3_results.csv",
    }[task]
    if not data_path.is_file():
        raise RuntimeError(f"Run experiments before plotting: {data_path}")
    with data_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in {data_path}")
    out_dir = PLOTS / task
    out_dir.mkdir(parents=True, exist_ok=True)
    configurations = list(dict.fromkeys(row["configuration"] for row in rows))
    traces = list(dict.fromkeys(row["trace"] for row in rows))
    metrics = ["ipc", "speedup", "l1d_load_mpki", "l2c_load_mpki", "l2i_load_mpki", "l2d_load_mpki"]
    if task in {"task1b", "task3"}:
        metrics += ["prefetch_accuracy_pct", "l1d_coverage_pct"]
    for metric in metrics:
        available = [row for row in rows if row.get(metric, "") not in {"", None}]
        if not available:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        for configuration in configurations:
            values = []
            labels = []
            for trace in traces:
                matches = [
                    row for row in available
                    if row["configuration"] == configuration and row["trace"] == trace
                ]
                if matches:
                    labels.append(trace)
                    values.append(float(matches[-1][metric]))
            if values:
                ax.plot(labels, values, marker="o", label=configuration)
        ax.set_title(f"{task}: {metric}")
        ax.set_xlabel("Trace")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}.png", dpi=180)
        plt.close(fig)
    print(f"Wrote plots to {out_dir.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "run", "plot", "all"])
    parser.add_argument("--task", choices=sorted(TASK_EXPERIMENTS), default="task1b")
    parser.add_argument("--traces", default="all", help="all, or comma-separated trace filenames")
    parser.add_argument("--warmup", type=positive_int, default=DEFAULT_WARMUP)
    parser.add_argument("--simulation", type=positive_int, default=DEFAULT_SIMULATION)
    parser.add_argument(
        "--thresholds",
        default="0.10,0.20,0.25,0.30,0.40",
        help="comma-separated dynamic I-RRIP thresholds",
    )
    args = parser.parse_args()
    if args.command == "list":
        for task, names in TASK_EXPERIMENTS.items():
            print(f"{task}:")
            for name in names:
                print(f"  {name}: {EXPERIMENTS[name].description}")
        return 0
    traces = traces_from_argument(args.traces)
    thresholds = [float(value) for value in args.thresholds.split(",") if value]
    if args.command in {"run", "all"}:
        rows = run_task(args.task, traces, args.warmup, args.simulation, thresholds)
        write_task_data(args.task, rows)
    if args.command in {"plot", "all"}:
        plot_task(args.task)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
