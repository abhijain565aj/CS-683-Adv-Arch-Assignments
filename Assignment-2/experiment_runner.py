#!/usr/bin/env python3
"""Build ChampSim experiments, run the assignment traces, and save metrics and plots."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "report"
DATA_DIR = REPORT / "data"
PLOTS_DIR = REPORT / "plots"
LOGS_DIR = REPORT / "logs"
DEFAULT_WARMUP = 25_000_000
DEFAULT_SIMULATION = 25_000_000
DEFAULT_THRESHOLDS = "0.10,0.20,0.25,0.30,0.40"
DEFAULT_FIXED_THRESHOLD = 0.25


@dataclass(frozen=True)
class Experiment:
    name: str
    prefetcher: str
    replacement: str
    description: str


# Names here must match the filenames in prefetcher/ and replacement/.
EXPERIMENTS = {
    "baseline": Experiment("baseline", "no", "lru", "No prefetching + LRU"),
    "ip_stride": Experiment("ip_stride", "ip_stride", "lru", "IP-stride + LRU"),
    "spp": Experiment("spp", "spp", "lru", "SPP + LRU"),
    "rrip": Experiment("rrip", "no", "rrip", "No prefetching + RRIP"),
    "irrip": Experiment("irrip", "no", "irrip", "No prefetching + I-RRIP"),
    "dynamic_irrip": Experiment(
        "dynamic_irrip", "no", "dyn_irrip", "No prefetching + Dynamic I-RRIP"
    ),
    "combined": Experiment(
        "combined", "spp", "prefetch_irrip", "SPP + prefetch-aware Dynamic I-RRIP"
    ),
}

# Task 2B is the only task with a five-threshold sensitivity sweep.
# Task 3 compares standalone SPP and Dynamic I-RRIP against their combination.
TASK_EXPERIMENTS = {
    "task1b": ("baseline", "ip_stride", "spp"),
    "task2a": ("rrip", "irrip"),
    "task2b": ("rrip", "irrip", "dynamic_irrip"),
    "task3": ("baseline", "spp", "dynamic_irrip", "combined"),
}

TASK_BASELINE = {
    "task1b": "baseline",
    "task2a": "rrip",
    "task2b": "rrip",
    "task3": "baseline",
}

CSV_PATHS = {
    "task1b": DATA_DIR / "task1b_results.csv",
    "task2a": DATA_DIR / "task2a_results.csv",
    "task2b": DATA_DIR / "task2b_sensitivity.csv",
    "task3": DATA_DIR / "task3_results.csv",
}

CSV_FIELDS = {
    "task1b": ["trace", "configuration", "ipc", "speedup", "l1d_load_mpki",
               "prefetch_accuracy_pct", "l1d_coverage_pct"],
    "task2a": ["trace", "configuration", "ipc", "speedup", "l2c_load_mpki",
               "l2i_load_mpki", "l2d_load_mpki"],
    "task2b": ["trace", "configuration", "threshold", "ipc", "speedup",
               "l2c_load_mpki", "l2i_load_mpki", "l2d_load_mpki"],
    "task3": ["trace", "configuration", "l1d_prefetcher", "l2_replacement", "threshold", "ipc", "speedup",
              "prefetch_accuracy_pct", "l1d_coverage_pct", "l1d_load_mpki",
              "l2c_load_mpki", "l2i_load_mpki", "l2d_load_mpki"],
}

# A no-prefetch configuration has no meaningful prefetch accuracy.
# Check only the metrics that its experiment actually needs.
REPORT_METRICS = {
    "task1b": ("ipc", "speedup", "l1d_load_mpki", "prefetch_accuracy_pct",
               "l1d_coverage_pct"),
    "task2a": ("ipc", "speedup", "l2c_load_mpki", "l2i_load_mpki",
               "l2d_load_mpki"),
    "task2b": ("ipc", "speedup", "l2c_load_mpki", "l2i_load_mpki",
               "l2d_load_mpki"),
    "task3": ("ipc", "speedup", "prefetch_accuracy_pct", "l1d_coverage_pct",
              "l1d_load_mpki", "l2c_load_mpki", "l2i_load_mpki",
              "l2d_load_mpki"),
}

METRIC_LABELS = {
    "ipc": "IPC",
    "speedup": "Speedup",
    "l1d_load_mpki": "L1D Load MPKI",
    "l2c_load_mpki": "L2C Load MPKI",
    "l2i_load_mpki": "L2 Instruction Load MPKI",
    "l2d_load_mpki": "L2 Data Load MPKI",
    "prefetch_accuracy_pct": "Prefetch accuracy (%)",
    "l1d_coverage_pct": "L1D coverage (%)",
}


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
PATTERNS = {
    "ipc": re.compile(rf"(?:Finished\s+)?CPU\s+0\b[^\n]*?cumulative IPC:\s*({NUMBER})", re.I),
    "l1d_load_mpki": re.compile(rf"\bL1D\s+LOAD\s+MPKI:\s*({NUMBER})", re.I),
    "l2c_load_mpki": re.compile(rf"\bL2C\s+LOAD\s+MPKI:\s*({NUMBER})", re.I),
    "l2i_load_mpki": re.compile(rf"\bL2C\s+INSTRUCTION\s+LOAD\s+MPKI:\s*({NUMBER})", re.I),
    "l2d_load_mpki": re.compile(rf"\bL2C\s+DATA\s+LOAD\s+MPKI:\s*({NUMBER})", re.I),
    "l1d_load_misses": re.compile(
        r"\bL1D\s+LOAD\b[^\n]*?\bMISS(?:ES)?\s*:\s*(\d+)", re.I
    ),
    "l2c_load_misses": re.compile(
        r"(?:^|\n)\s*L2C\s+LOAD\b[^\n]*?\bMISS(?:ES)?\s*:\s*(\d+)", re.I
    ),
    "prefetch_useful": re.compile(r"\bL1D\s+USEFUL\s+LOAD\s+PREFETCHES:\s*(\d+)", re.I),
    "prefetch_issued": re.compile(
        r"\bL1D\s+USEFUL\s+LOAD\s+PREFETCHES:[^\n]*?"
        r"\bPREFETCH\s+ISSUED\s+TO\s+LOWER\s+LEVEL:\s*(\d+)", re.I
    ),
}
ACCURACY_PATTERN = re.compile(
    rf"\bL1D\s+USEFUL\s+LOAD\s+PREFETCHES:[^\n]*?\bACCURACY:\s*({NUMBER})", re.I
)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def threshold_value(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1")
    return number


def parse_thresholds(value: str) -> list[float]:
    try:
        thresholds = [threshold_value(x.strip()) for x in value.split(",") if x.strip()]
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise ValueError(f"Invalid threshold list: {value}") from exc
    if not thresholds or len(thresholds) != len(set(thresholds)):
        raise ValueError("Supply a nonempty list of distinct thresholds")
    return thresholds


def parse_output(output: str, simulation_instructions: Optional[int] = None) -> dict[str, float]:
    result = {}
    for key, pattern in PATTERNS.items():
        matches = pattern.findall(output)
        if matches:
            result[key] = float(matches[-1])

    # This ChampSim fork reports L1D and L2C LOAD MISS counts but may omit
    # aggregate LOAD MPKI. Derive it from the simulation instruction count.
    # Never use the number of warmup instructions in the denominator.
    if simulation_instructions and simulation_instructions > 0:
        for prefix in ("l1d", "l2c"):
            name = f"{prefix}_load_mpki"
            misses = result.get(f"{prefix}_load_misses")
            if name not in result and misses is not None:
                result[name] = 1000.0 * misses / simulation_instructions
                result[f"{name}_source"] = "computed from LOAD misses"

        # Some forks only print instruction and data load MPKI at L2, rather
        # than an aggregate L2C LOAD line. Both components have the same
        # denominator and their sum gives the corresponding total.
        if ("l2c_load_mpki" not in result
                and "l2i_load_mpki" in result and "l2d_load_mpki" in result):
            result["l2c_load_mpki"] = (
                result["l2i_load_mpki"] + result["l2d_load_mpki"]
            )
            result["l2c_load_mpki_source"] = "instruction + data LOAD MPKI"

    # Prefer accuracy from the simulator's actual issued/useful counters.
    issued = result.get("prefetch_issued")
    useful = result.get("prefetch_useful")
    if issued is not None and useful is not None:
        result["prefetch_accuracy_pct"] = (100 * useful / issued) if issued else 0.0
    else:
        matches = ACCURACY_PATTERN.findall(output)
        if matches:
            result["prefetch_accuracy_pct"] = float(matches[-1])
    return result


def find_traces(value: str, traces_dir: Optional[Path]) -> list[Path]:
    if traces_dir is None:
        traces_dir = ROOT.parent / "traces"
        if not traces_dir.is_dir():
            traces_dir = ROOT / "traces"
    elif not traces_dir.is_absolute():
        traces_dir = (ROOT / traces_dir).resolve()

    if value == "all":
        # The provided assignment traces can be Tr1.gz / Tr2.gz / Tr3.gz,
        # not necessarily *.champsimtrace.xz.
        files = sorted(
            p for p in traces_dir.iterdir()
            if p.is_file() and (
                p.name.endswith((".gz", ".xz", ".champsimtrace"))
            )
        ) if traces_dir.is_dir() else []
    else:
        files = []
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            path = Path(item)
            if path.is_absolute():
                candidate = path
            elif path.is_file():
                candidate = path.resolve()
            else:
                candidate = traces_dir / path
            if not candidate.is_file():
                raise RuntimeError(f"Trace not found: {candidate}")
            files.append(candidate)

    if not files:
        raise RuntimeError(f"No ChampSim traces found in {traces_dir}")
    return files


def verify_sources(experiment: Experiment) -> None:
    required = (
        ROOT / "prefetcher" / f"{experiment.prefetcher}.l1d_pref",
        ROOT / "replacement" / f"{experiment.replacement}.l2c_repl",
    )
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"Missing file for {experiment.name}: {path}")


def execute(command: list[str], log_path: Path, env: Optional[dict] = None,
            debug: bool = False) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$", shlex.join(command), flush=True)
    output_lines = []
    with log_path.open("w") as handle:
        # A brief retry handles ETXTBSY when ChampSim has just replaced an executable.
        for attempt in range(3):
            try:
                process = subprocess.Popen(
                    command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                )
                break
            except OSError as exc:
                if exc.errno != 26 or attempt == 2:
                    raise
                time.sleep(1)
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            output_lines.append(line)
            if debug:
                print(line, end="", flush=True)
        return_code = process.wait()
    if return_code:
        tail = "".join(output_lines[-45:]).rstrip()
        raise RuntimeError(
            f"Command failed (exit {return_code}). Log: {log_path}"
            + (f"\n\nLast 45 lines of command output:\n{tail}" if tail else "")
        )
    return "".join(output_lines)


def build_experiment(experiment: Experiment, task: str, debug: bool) -> Path:
    verify_sources(experiment)
    binary_name = f"runner-{experiment.name}"
    print(f"\nBuilding {experiment.name}: {experiment.description}", flush=True)
    execute(
        ["bash", str(ROOT / "build_champsim.sh"),
         experiment.prefetcher, experiment.replacement, binary_name],
        LOGS_DIR / task / f"build-{experiment.name}.log", debug=debug,
    )
    binaries = sorted((ROOT / "bin").glob(f"*-{binary_name}"))
    if not binaries:
        raise RuntimeError(f"Build finished, but no binary *-{binary_name} found")
    return binaries[-1].resolve()


def run_log_path(experiment: Experiment, task: str, trace: Path,
                 threshold: Optional[float]) -> Path:
    trace_id = re.sub(r"\.champsimtrace(?:\.xz|\.gz)?$", "", trace.name)
    label = f"{experiment.name}-{trace_id}"
    if experiment.name == "combined":
        # Keep runs with different combined policies from sharing saved logs.
        label = f"combined-{experiment.replacement}-{trace_id}"
    if threshold is not None:
        label += f"-threshold-{threshold:g}"
    return LOGS_DIR / task / f"{label}.log"


def reuse_completed_run(experiment: Experiment, task: str, trace: Path,
                        warmup: int, simulation: int,
                        threshold: Optional[float]) -> Optional[dict]:
    """Reuse a successful-looking log only when --resume is explicitly requested."""
    log_path = run_log_path(experiment, task, trace, threshold)
    if not log_path.is_file():
        return None
    metrics = parse_output(log_path.read_text(errors="replace"), simulation)
    if "ipc" not in metrics:
        return None
    print(f"Reusing {log_path.relative_to(ROOT)}", flush=True)
    return {
        "trace": trace.name,
        "configuration": experiment.name,
        "description": experiment.description,
        "l1d_prefetcher": experiment.prefetcher,
        "l2_replacement": experiment.replacement,
        "threshold": threshold,
        "warmup_instructions": warmup,
        "simulation_instructions": simulation,
        "log": str(log_path.relative_to(ROOT)),
        **metrics,
    }


def run_one(experiment: Experiment, binary: Path, task: str, trace: Path,
            warmup: int, simulation: int, threshold: Optional[float],
            debug: bool) -> dict:
    log_path = run_log_path(experiment, task, trace, threshold)

    env = os.environ.copy()
    # Explicitly remove an inherited threshold for policies that should use their defaults.
    env.pop("CHAMPSIM_THRESHOLD", None)
    if threshold is not None:
        env["CHAMPSIM_THRESHOLD"] = str(threshold)

    print(f"\nRunning {task}: {experiment.name} | {trace.name}"
          f" | threshold={threshold if threshold is not None else '-'}", flush=True)
    output = execute(
        [str(binary), "-warmup_instructions", str(warmup),
         "-simulation_instructions", str(simulation), "-traces", str(trace)],
        log_path, env=env, debug=debug,
    )
    metrics = parse_output(output, simulation)
    if "ipc" not in metrics:
        raise RuntimeError(f"IPC not found in {log_path}; check the simulator output")

    return {
        "trace": trace.name,
        "configuration": experiment.name,
        "description": experiment.description,
        "l1d_prefetcher": experiment.prefetcher,
        "l2_replacement": experiment.replacement,
        "threshold": threshold,
        "warmup_instructions": warmup,
        "simulation_instructions": simulation,
        "log": str(log_path.relative_to(ROOT)),
        **metrics,
    }


def add_derived_metrics(rows: list[dict], task: str) -> None:
    baselines = {
        row["trace"]: row
        for row in rows
        if row["configuration"] == TASK_BASELINE[task]
    }
    for row in rows:
        base = baselines.get(row["trace"])
        if base is None:
            continue
        base_ipc = base.get("ipc")
        if base_ipc:
            row["speedup"] = row["ipc"] / base_ipc

        # Prefetch coverage is defined relative to the no-prefetch LRU baseline.
        # It is not meaningful to attribute a change from replacement alone to prefetching.
        if task in ("task1b", "task3") and row["configuration"] in (
            "baseline", "ip_stride", "spp", "combined"
        ):
            baseline_misses = base.get("l1d_load_misses")
            current_misses = row.get("l1d_load_misses")
            if baseline_misses and current_misses is not None:
                row["l1d_coverage_pct"] = (
                    100.0 * (baseline_misses - current_misses) / baseline_misses
                )


def write_task_data(task: str, rows: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = CSV_PATHS[task]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS[task], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (DATA_DIR / f"{task}_raw.json").write_text(json.dumps(rows, indent=2))
    print(f"Saved {path.relative_to(ROOT)} ({len(rows)} rows)")

    # Save a readable table in addition to CSV/JSON. Missing values remain blank.
    table_path = DATA_DIR / f"{task}_summary.md"
    fields = CSV_FIELDS[task]
    with table_path.open("w") as handle:
        handle.write(f"# {task.upper()} experiment results\n\n")
        handle.write("| " + " | ".join(fields) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
        for row in rows:
            cells = [f"{row[field]:.5f}" if isinstance(row.get(field), float)
                     else str(row.get(field, "") if row.get(field) is not None else "")
                     for field in fields]
            handle.write("| " + " | ".join(cells) + " |\n")
    print(f"Saved {table_path.relative_to(ROOT)}")

    # The CSV is still written when a metric cannot be parsed. Never invent values.
    missing_by_row = []
    for row in rows:
        required = set(REPORT_METRICS[task])
        if row["configuration"] in {"baseline", "rrip", "irrip", "dynamic_irrip"}:
            required.discard("prefetch_accuracy_pct")
        if row["configuration"] in {"rrip", "irrip", "dynamic_irrip"}:
            required.discard("l1d_coverage_pct")
        absent = sorted(field for field in required if row.get(field) is None)
        if absent:
            missing_by_row.append((row, absent))
    if missing_by_row:
        print(f"WARNING: Missing metric(s) in {len(missing_by_row)} experiment row(s):",
              file=sys.stderr)
        for row, absent in missing_by_row[:20]:
            print(f"  {row['trace']} | {row['configuration']} | "
                  f"threshold={row.get('threshold')} | {', '.join(absent)}; "
                  f"see {row['log']}", file=sys.stderr)
        if len(missing_by_row) > 20:
            print(f"  ... and {len(missing_by_row) - 20} more row(s)", file=sys.stderr)
    else:
        print("All applicable report metrics were extracted.")


def run_task(task: str, traces: list[Path], warmup: int, simulation: int,
             thresholds: list[float], fixed_threshold: float, debug: bool,
             resume: bool = False) -> list[dict]:
    rows = []
    for name in TASK_EXPERIMENTS[task]:
        experiment = EXPERIMENTS[name]
        if task == "task2b" and name == "dynamic_irrip":
            selected_thresholds = thresholds
        elif task == "task3" and name in ("dynamic_irrip", "combined"):
            selected_thresholds = [fixed_threshold]
        else:
            selected_thresholds = [None]

        binary = None  # Build only if at least one run isn't reusable.
        for threshold in selected_thresholds:
            for trace in traces:
                if resume:
                    previous = reuse_completed_run(
                        experiment, task, trace, warmup, simulation, threshold
                    )
                    if previous is not None:
                        rows.append(previous)
                        continue
                if binary is None:
                    binary = build_experiment(experiment, task, debug)
                rows.append(run_one(experiment, binary, task, trace, warmup,
                                    simulation, threshold, debug))

    add_derived_metrics(rows, task)
    return rows



def reparse_task(task: str) -> None:
    """Rebuild CSV/plots from saved ChampSim logs without another simulation."""
    raw_path = DATA_DIR / f"{task}_raw.json"
    if not raw_path.is_file():
        raise RuntimeError(f"Missing {raw_path}; run this task first")
    rows = json.loads(raw_path.read_text())
    if not rows:
        raise RuntimeError(f"No experiment rows in {raw_path}")
    for row in rows:
        log_path = ROOT / row["log"]
        if not log_path.is_file():
            raise RuntimeError(f"Missing simulator log: {log_path}")
        output = log_path.read_text(errors="replace")
        # Remove stale parser outputs but keep configuration and run metadata.
        for key in list(row):
            if key not in {"trace", "configuration", "description", "threshold",
                           "l1d_prefetcher", "l2_replacement",
                           "warmup_instructions", "simulation_instructions", "log"}:
                del row[key]
        metrics = parse_output(output, int(row["simulation_instructions"]))
        if "ipc" not in metrics:
            raise RuntimeError(f"IPC not found in simulator log {log_path}")
        row.update(metrics)
    add_derived_metrics(rows, task)
    write_task_data(task, rows)
    plot_task(task)


def save_figure(fig, output_base: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=180)
    fig.savefig(output_base.with_suffix(".pdf"))


def plot_task(task: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")  # Works in Docker/headless environments.
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to generate plots") from exc

    path = CSV_PATHS[task]
    if not path.is_file():
        raise RuntimeError(f"Run the task before plotting: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in {path}")

    output_dir = PLOTS_DIR / task
    output_dir.mkdir(parents=True, exist_ok=True)
    traces = list(dict.fromkeys(row["trace"] for row in rows))
    plotted = []

    # Report comparison charts: one curve per configuration and threshold.
    for metric in REPORT_METRICS[task]:
        fig, ax = plt.subplots(figsize=(10, 5))
        series = {}
        for row in rows:
            raw = row.get(metric)
            if raw in (None, ""):
                continue
            label = row["configuration"]
            if task == "task2b" and label == "dynamic_irrip":
                label += f" (threshold={float(row['threshold']):g})"
            series.setdefault(label, {})[row["trace"]] = float(raw)
        for label, data in series.items():
            xs = [trace for trace in traces if trace in data]
            if xs:
                ax.plot(xs, [data[x] for x in xs], marker="o", label=label)
        if series:
            ax.set_xlabel("Trace")
            ax.set_ylabel(METRIC_LABELS[metric])
            ax.set_title(f"{task.upper()} — {METRIC_LABELS[metric]}")
            ax.tick_params(axis="x", labelrotation=30)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            save_figure(fig, output_dir / metric)
            plotted.append(metric)
        else:
            print(f"WARNING: No values of {metric} for {task}; plot skipped",
                  file=sys.stderr)
        plt.close(fig)

    # For the threshold sensitivity study, also plot metric vs threshold,
    # separately for each trace. The fixed RRIP and I-RRIP runs are not
    # included in these plots because their policy has no threshold.
    if task == "task2b":
        dynamic = [row for row in rows if row["configuration"] == "dynamic_irrip"
                   and row.get("threshold") not in (None, "")]
        for metric in REPORT_METRICS[task]:
            fig, ax = plt.subplots(figsize=(9, 5))
            found = False
            for trace in traces:
                pairs = sorted(
                    (float(row["threshold"]), float(row[metric]))
                    for row in dynamic
                    if row["trace"] == trace and row.get(metric) not in (None, "")
                )
                if pairs:
                    ax.plot([x for x, _ in pairs], [y for _, y in pairs],
                            marker="o", label=trace)
                    found = True
            if found:
                ax.set_xlabel("Dynamic I-RRIP threshold")
                ax.set_ylabel(METRIC_LABELS[metric])
                ax.set_title(f"Task 2B sensitivity — {METRIC_LABELS[metric]}")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc="best")
                save_figure(fig, output_dir / f"sensitivity_{metric}")
                plotted.append(f"sensitivity_{metric}")
            else:
                print(f"WARNING: No sensitivity data for {metric}", file=sys.stderr)
            plt.close(fig)

    print(f"Saved {len(plotted)} plot pairs (PNG and PDF) under "
          f"{output_dir.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "run", "plot", "all", "reparse"))
    parser.add_argument("--task", choices=tuple(TASK_EXPERIMENTS), default="task1b")
    parser.add_argument("--traces", default="all", help="all or comma-separated filenames/paths")
    parser.add_argument("--traces-dir", type=Path, default=None,
                        help="trace directory (default: ../traces, or ./traces)")
    parser.add_argument("--warmup", type=positive_int, default=DEFAULT_WARMUP)
    parser.add_argument("--simulation", type=positive_int, default=DEFAULT_SIMULATION)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS,
                        help="Task 2B only: comma-separated thresholds")
    parser.add_argument("--fixed-threshold", type=threshold_value,
                        default=DEFAULT_FIXED_THRESHOLD,
                        help="Task 3 threshold, shared by dynamic and combined configurations")
    parser.add_argument("--combined-replacement", default="prefetch_irrip",
                        help="Task 3 combined L2 policy (default: dyn_irrip as specified in PA2; "
                             "use prefetch_irrip only if that separate policy builds)")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse existing simulation logs with a parsed IPC; "
                             "use only when the underlying C++ implementations have not changed")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.task == "task3":
        name = args.combined_replacement
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            parser.error("--combined-replacement must be a replacement policy filename stem")
        description = (
            "SPP + prefetch-aware Dynamic I-RRIP"
            if name == "prefetch_irrip" else f"SPP + {name}"
        )
        EXPERIMENTS["combined"] = Experiment("combined", "spp", name, description)

    if args.command == "list":
        for task, names in TASK_EXPERIMENTS.items():
            print(f"{task}:")
            for name in names:
                x = EXPERIMENTS[name]
                print(f"  {name}: {x.prefetcher}/{x.replacement} - {x.description}")
        return 0

    if args.command == "reparse":
        reparse_task(args.task)
        return 0

    if args.command in ("run", "all"):
        traces = find_traces(args.traces, args.traces_dir)
        thresholds = parse_thresholds(args.thresholds) if args.task == "task2b" else []
        rows = run_task(args.task, traces, args.warmup, args.simulation,
                        thresholds, args.fixed_threshold, args.debug,
                        resume=args.resume)
        write_task_data(args.task, rows)
    if args.command in ("plot", "all"):
        plot_task(args.task)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
