#!/usr/bin/env python3
"""Build ChampSim experiments, run the assignment traces, and save metrics and plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    "task3": ["trace", "configuration", "l1d_prefetcher", "l2_replacement", "threshold",
              "ipc", "speedup", "prefetch_accuracy_pct", "l1d_coverage_pct",
              "l1d_load_mpki", "l2c_load_mpki", "l2i_load_mpki", "l2d_load_mpki"],
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
             resume: bool = False, stage: Optional[str] = None) -> list[dict]:
    """Run every task configuration, or just one named configuration via --stage."""
    task_stages = TASK_EXPERIMENTS[task]
    if stage is not None and stage not in task_stages:
        valid = ", ".join(task_stages)
        raise ValueError(f"Stage '{stage}' is not part of {task}. Valid stages: {valid}")

    selected_stages = (stage,) if stage is not None else task_stages
    rows = []
    for name in selected_stages:
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



def _row_key(row: dict, task: str) -> tuple[str, str, Optional[str]]:
    """Stable identity for one trace/configuration and (for Task 2B) threshold."""
    # Task 2B intentionally stores multiple threshold runs for each trace. Task 3
    # uses one selected fixed threshold per configuration, so a targeted rerun at
    # a new fixed threshold should replace that configuration's previous row.
    threshold = row.get("threshold") if task == "task2b" else None
    if threshold in (None, ""):
        threshold_key = None
    else:
        try:
            threshold_key = f"{float(threshold):.12g}"
        except (TypeError, ValueError):
            threshold_key = str(threshold)
    return (str(row.get("trace", "")), str(row.get("configuration", "")), threshold_key)


def _load_existing_task_rows(task: str) -> list[dict]:
    """Load existing JSON results; fall back to CSV if JSON is absent."""
    raw_path = DATA_DIR / f"{task}_raw.json"
    if raw_path.is_file():
        loaded = json.loads(raw_path.read_text())
        if not isinstance(loaded, list):
            raise RuntimeError(f"Expected a list of result rows in {raw_path}")
        return loaded

    csv_path = CSV_PATHS[task]
    if not csv_path.is_file():
        return []
    with csv_path.open(newline="") as handle:
        loaded = list(csv.DictReader(handle))
    numeric_fields = set(CSV_FIELDS[task]) - {"trace", "configuration", "l1d_prefetcher", "l2_replacement"}
    for row in loaded:
        for field in numeric_fields:
            value = row.get(field)
            if value in (None, ""):
                row[field] = None
                continue
            try:
                row[field] = float(value)
            except (TypeError, ValueError):
                pass
        if row.get("threshold") is None:
            row["threshold"] = None
    return loaded


def merge_task_rows(task: str, new_rows: list[dict]) -> list[dict]:
    """Merge targeted reruns into saved results, replacing only matching row keys."""
    merged = _load_existing_task_rows(task)
    positions = {_row_key(row, task): index for index, row in enumerate(merged)}
    for row in new_rows:
        key = _row_key(row, task)
        if key in positions:
            merged[positions[key]] = row
        else:
            positions[key] = len(merged)
            merged.append(row)
    # Derive speedup/coverage after merging so they can use the preserved baseline row.
    add_derived_metrics(merged, task)
    return merged


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
            if key not in {"trace", "configuration", "description", "l1d_prefetcher",
                           "l2_replacement", "threshold", "warmup_instructions",
                           "simulation_instructions", "log"}:
                del row[key]
        metrics = parse_output(output, int(row["simulation_instructions"]))
        if "ipc" not in metrics:
            raise RuntimeError(f"IPC not found in simulator log {log_path}")
        row.update(metrics)
    add_derived_metrics(rows, task)
    write_task_data(task, rows)
    plot_task(task)


def save_figure(fig, output_base: Path) -> None:
    """Save high-resolution PNG plots only (no PDF sidecars)."""
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight")


def plot_task(task: str) -> None:
    """Plot measured results with data-aware axes and a few task-specific effect charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
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
    trace_labels = []
    for trace in traces:
        name = Path(trace).name
        for suffix in (".champsimtrace.xz", ".champsimtrace.gz", ".champsimtrace",
                       ".trace.xz", ".trace.gz", ".trace", ".gz", ".xz"):
            if name.lower().endswith(suffix):
                name = name[:-len(suffix)]
                break
        match = re.fullmatch(r"(?i:tr(?:ace)?)[_ -]?(\d+)", name)
        trace_labels.append(f"Trace {int(match.group(1))}" if match else name)

    task_titles = {
        "task1b": "Task 1B — Stride Pattern Prefetching (SPP)",
        "task2a": "Task 2A — Instruction-Conscious L2 Replacement (I-RRIP)",
        "task2b": "Task 2B — Dynamic I-RRIP",
        "task3": "Task 3 — SPP + Prefetch-Aware Dynamic I-RRIP",
    }
    config_labels = {
        "baseline": "Baseline (No Prefetch)",
        "ip_stride": "IP-stride",
        "spp": "SPP",
        "rrip": "RRIP",
        "irrip": "I-RRIP",
        "dynamic_irrip": "Dynamic I-RRIP",
        "combined": "SPP + Prefetch-aware Dynamic I-RRIP",
    }
    config_colors = {
        "baseline": "#8C8C8C",
        "ip_stride": "#4C72B0",
        "spp": "#55A868",
        "rrip": "#4C72B0",
        "irrip": "#DD8452",
        "dynamic_irrip": "#8172B2",
        "combined": "#64B5CD",
    }
    threshold_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    task_metric_titles = {
        "ipc": "Instructions Per Cycle (IPC)",
        "speedup": "Speedup Relative to Task Baseline",
        "l1d_load_mpki": "L1 Data Cache Load MPKI",
        "l2c_load_mpki": "L2 Cache Load MPKI",
        "l2i_load_mpki": "L2 Instruction Load MPKI",
        "l2d_load_mpki": "L2 Data Load MPKI",
        "prefetch_accuracy_pct": "L1D Prefetch Accuracy",
        "l1d_coverage_pct": "L1D Load-Miss Coverage",
    }

    def get_number(row: dict, metric: str) -> Optional[float]:
        raw = row.get(metric)
        if raw in (None, ""):
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def metric_label(metric: str) -> str:
        return METRIC_LABELS.get(metric, metric.replace("_", " ").title())

    def metric_title(metric: str) -> str:
        return task_metric_titles.get(metric, metric_label(metric))

    def format_value(metric: str, value: float) -> str:
        if metric in ("prefetch_accuracy_pct", "l1d_coverage_pct"):
            return f"{value:.1f}%"
        if metric in ("ipc",):
            return f"{value:.3f}"
        if metric == "speedup":
            return f"{value:.4f}"
        return f"{value:.2f}"

    def nice_step(value: float, target_ticks: int = 6) -> float:
        """Return a readable tick step near value / target_ticks."""
        if value <= 0:
            return 1.0
        raw = value / target_ticks
        power = 10.0 ** math.floor(math.log10(raw))
        fraction = raw / power
        choices = (1.0, 2.0, 2.5, 5.0, 10.0)
        chosen = min(choices, key=lambda candidate: abs(math.log(candidate / fraction)))
        return chosen * power

    def nice_ceil(value: float, target_ticks: int = 6) -> float:
        step = nice_step(value, target_ticks)
        return max(step, math.ceil((value - 1e-12) / step) * step)

    def nice_floor(value: float, target_ticks: int = 6) -> float:
        step = nice_step(abs(value), target_ticks)
        return math.floor((value + 1e-12) / step) * step

    def style_axes(ax, metric: str, values: list[float]) -> None:
        finite = [v for v in values if v is not None]
        if not finite:
            finite = [0.0]
        maximum = max(finite)
        minimum = min(finite)
        if metric == "speedup":
            # Speedup is a ratio centred on 1.0, so bars start at the no-change
            # reference, and the explicitly labelled zoomed axis shows small effects.
            deviation = max([abs(v - 1.0) for v in finite] + [0.002])
            padding = max(0.0015, deviation * 0.22)
            ax.set_ylim(min(minimum, 1.0) - padding, max(maximum, 1.0) + padding)
            ax.axhline(1.0, color="#B44A4A", linestyle="--", linewidth=1.25,
                       label="No speedup (1.0)", zorder=2)
            ax.set_ylabel("Speedup ratio (1.0 = no change; zoomed axis)")
        elif metric in ("prefetch_accuracy_pct", "l1d_coverage_pct"):
            # Percentage plots use the observed range, not a fixed 0–100% range.
            # This helps distinguish values such as 0.3% from 1.2% while keeping
            # the natural zero reference and never exceeding 100% on the top end.
            positive_max = max(0.0, maximum)
            upper = min(100.0, nice_ceil(max(positive_max * 1.10, 0.01)))
            lower = 0.0 if minimum >= 0 else nice_floor(minimum * 1.15)
            ax.set_ylim(lower, max(upper, lower + 0.01))
            if metric == "l1d_coverage_pct":
                ax.axhline(0, color="#555555", linewidth=0.8, zorder=2)
            ax.set_ylabel(metric_label(metric))
        else:
            positive_max = max(0.0, maximum)
            upper = nice_ceil(max(positive_max * 1.10, 0.01))
            lower = min(0.0, nice_floor(minimum * 1.10)) if minimum < 0 else 0.0
            ax.set_ylim(lower, max(upper, lower + 0.01))
            ax.set_ylabel(metric_label(metric))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
        ax.grid(axis="y", linestyle="--", alpha=0.28, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def add_bar_labels(ax, containers, metric: str, enabled: bool = True) -> None:
        if not enabled:
            return
        for container in containers:
            for bar in container:
                if metric == "speedup":
                    # Speedup bars are deviations from 1.0 but labels show the ratio.
                    endpoint = bar.get_y() + bar.get_height()
                    label_value = endpoint
                    positive = bar.get_height() >= 0
                    y = endpoint
                else:
                    label_value = bar.get_height()
                    y = bar.get_y() + bar.get_height()
                    positive = bar.get_height() >= 0
                offset = 2 if positive else -2
                ax.annotate(
                    format_value(metric, label_value),
                    xy=(bar.get_x() + bar.get_width() / 2, y),
                    xytext=(0, offset), textcoords="offset points",
                    ha="center", va="bottom" if positive else "top",
                    fontsize=7, clip_on=False,
                )

    def grouped_bars(ax, x_positions, categories, series, metric: str,
                     colors, category_axis_label: str, label_values: bool = True):
        """series is an ordered list of (legend label, values keyed by category)."""
        count = max(1, len(series))
        cluster_width = 0.82
        bar_width = cluster_width / count
        containers, plotted_values = [], []
        for series_index, (label, values_by_category) in enumerate(series):
            offset = (series_index - (count - 1) / 2) * bar_width
            xs, ys = [], []
            for index, category in enumerate(categories):
                value = values_by_category.get(category)
                if value is None:
                    continue
                xs.append(x_positions[index] + offset)
                ys.append(value)
                plotted_values.append(value)
            if not xs:
                continue
            color = colors[series_index % len(colors)]
            if metric == "speedup":
                container = ax.bar(xs, [value - 1.0 for value in ys], bottom=1.0,
                                   width=bar_width * 0.94, label=label, color=color,
                                   edgecolor="white", linewidth=0.45, zorder=3)
            else:
                container = ax.bar(xs, ys, width=bar_width * 0.94, label=label,
                                   color=color, edgecolor="white", linewidth=0.45,
                                   zorder=3)
            containers.append(container)
        add_bar_labels(ax, containers, metric, enabled=label_values)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(categories)
        ax.set_xlabel(category_axis_label)
        style_axes(ax, metric, plotted_values)
        return containers, plotted_values

    def color_for_label(label: str) -> str:
        if label.startswith("Baseline"):
            config = "baseline"
        elif label == "IP-stride":
            config = "ip_stride"
        elif label == "SPP":
            config = "spp"
        elif label == "RRIP":
            config = "rrip"
        elif label == "I-RRIP":
            config = "irrip"
        elif label.startswith("Dynamic I-RRIP"):
            config = "dynamic_irrip"
        elif label.startswith("SPP +"):
            config = "combined"
        else:
            config = "baseline"
        return config_colors.get(config, "#4C72B0")

    def prepare_comparison_series(metric: str):
        task_rows = rows
        ref_threshold = None
        if task == "task2b":
            available = sorted({
                float(row["threshold"]) for row in rows
                if row.get("configuration") == "dynamic_irrip"
                and row.get("threshold") not in (None, "")
            })
            if available:
                ref_threshold = min(available, key=lambda value: abs(value - DEFAULT_FIXED_THRESHOLD))
            task_rows = [
                row for row in rows
                if row.get("configuration") in ("rrip", "irrip")
                or (row.get("configuration") == "dynamic_irrip"
                    and ref_threshold is not None
                    and row.get("threshold") not in (None, "")
                    and abs(float(row["threshold"]) - ref_threshold) < 1e-9)
            ]

        order = ("rrip", "irrip", "dynamic_irrip") if task == "task2b" else TASK_EXPERIMENTS[task]
        assembled = []
        for config in order:
            # Accuracy is undefined when a configuration issues no prefetches.
            if metric == "prefetch_accuracy_pct" and config in {
                "baseline", "rrip", "irrip", "dynamic_irrip"
            }:
                continue
            config_rows = [row for row in task_rows if row.get("configuration") == config]
            values_by_trace = {}
            for row in config_rows:
                value = get_number(row, metric)
                if value is not None:
                    values_by_trace[row["trace"]] = value
            if not values_by_trace:
                continue
            label = config_labels.get(config, config)
            if task == "task2b" and config == "dynamic_irrip" and ref_threshold is not None:
                label = f"Dynamic I-RRIP (threshold={ref_threshold:g})"
            assembled.append((label, values_by_trace))
        return assembled, ref_threshold

    def value_by_config_trace(configuration: str, metric: str, trace: str) -> Optional[float]:
        candidates = [row for row in rows if row.get("configuration") == configuration
                      and row.get("trace") == trace]
        if task == "task2b" and configuration == "dynamic_irrip":
            candidates = [row for row in candidates if row.get("threshold") not in (None, "")]
            if not candidates:
                return None
            chosen = min(candidates, key=lambda row: abs(float(row["threshold"]) - DEFAULT_FIXED_THRESHOLD))
            return get_number(chosen, metric)
        return get_number(candidates[0], metric) if candidates else None

    def plot_effect_chart(filename: str, title: str, ylabel: str, categories: list[str],
                          series: list[tuple[str, dict[str, float]]], colors: list[str],
                          label_values: bool = True) -> None:
        """Draw changes/reductions around zero; unlike absolute bars, effects are directly comparable."""
        if not series:
            return
        fig_width = max(8.8, 2.1 * len(categories) + 0.8 * len(series))
        fig, ax = plt.subplots(figsize=(fig_width, 5.2))
        count = len(series)
        cluster_width = 0.78
        bar_width = cluster_width / count
        all_values = []
        containers = []
        for s_idx, (label, vals) in enumerate(series):
            xs, ys = [], []
            offset = (s_idx - (count - 1) / 2) * bar_width
            for i, category in enumerate(categories):
                value = vals.get(category)
                if value is None:
                    continue
                xs.append(i + offset)
                ys.append(value)
                all_values.append(value)
            if xs:
                color = colors[s_idx % len(colors)]
                cont = ax.bar(xs, ys, width=bar_width * 0.93, label=label,
                              color=color, edgecolor="white", linewidth=0.45, zorder=3)
                containers.append(cont)
        if not all_values:
            plt.close(fig)
            return
        ax.axhline(0, color="#555555", linewidth=1.0, zorder=2)
        minimum, maximum = min(all_values), max(all_values)
        # Keep zero visible, but do not waste half the chart when every effect
        # is beneficial (or detrimental). If signs are mixed, preserve room for both.
        if minimum >= 0:
            lower = 0.0
            upper = nice_ceil(max(maximum * 1.15, 0.01), target_ticks=5)
        elif maximum <= 0:
            lower = nice_floor(minimum * 1.15, target_ticks=5)
            upper = 0.0
        else:
            lower = nice_floor(minimum * 1.15, target_ticks=5)
            upper = nice_ceil(maximum * 1.15, target_ticks=5)
        ax.set_ylim(lower, upper if upper > lower else lower + 0.01)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
        ax.grid(axis="y", linestyle="--", alpha=0.28, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(range(len(categories))))
        ax.set_xticklabels(categories)
        ax.set_xlabel("Benchmark trace" if categories and categories[0].startswith("Trace") else "Decision threshold")
        ax.set_title(title, fontsize=12, fontweight="semibold", pad=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if label_values:
            for cont in containers:
                for bar in cont:
                    val = bar.get_height()
                    label = f"{val:+.2f}%" if abs(val) < 1 else f"{val:+.1f}%"
                    ax.annotate(label,
                                xy=(bar.get_x() + bar.get_width() / 2, val),
                                xytext=(0, 2 if val >= 0 else -2),
                                textcoords="offset points", ha="center",
                                va="bottom" if val >= 0 else "top", fontsize=7,
                                clip_on=False)
        if len(series) > 1:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                      ncol=min(3, len(series)), frameon=False, fontsize=8.5)
            fig.tight_layout(rect=(0, 0.08, 1, 1))
        else:
            fig.tight_layout()
        save_figure(fig, output_dir / filename)
        plt.close(fig)

    plotted = []
    # Required absolute metric charts. Percentage axes scale to the observed data range;
    # speedup bars are drawn around 1.0 so small differences remain visible and explicit.
    for metric in REPORT_METRICS[task]:
        series, ref_threshold = prepare_comparison_series(metric)
        if not series:
            print(f"WARNING: No values of {metric} for {task}; plot skipped", file=sys.stderr)
            continue
        labels = [label for label, _ in series]
        categories = [trace_labels[index] for index in range(len(traces))]
        pretty_series = []
        for label, values_by_trace in series:
            pretty_series.append((label, {
                trace_labels[index]: values_by_trace[trace]
                for index, trace in enumerate(traces) if trace in values_by_trace
            }))
        fig_width = max(9.5, 2.25 * len(categories) + 0.8 * len(labels))
        fig, ax = plt.subplots(figsize=(fig_width, 5.7))
        colors = [color_for_label(label) for label in labels]
        grouped_bars(ax, list(range(len(categories))), categories, pretty_series, metric,
                     colors, "Benchmark trace", label_values=(len(labels) <= 4))
        title = f"{task_titles[task]}\n{metric_title(metric)} by Configuration"
        if metric == "speedup":
            title += " (zoomed around 1.0; 1.0 = no change)"
        if task == "task2b" and ref_threshold is not None:
            title += f" (reference threshold = {ref_threshold:g})"
        ax.set_title(title, fontsize=12, fontweight="semibold", pad=12)
        ax.margins(x=0.04)
        # Keep the reference-line legend entry in speedup charts as well.
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17),
                  ncol=min(3, len(labels) + (1 if metric == "speedup" else 0)),
                  frameon=False, fontsize=8.5)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        save_figure(fig, output_dir / metric)
        plotted.append(metric)
        plt.close(fig)

    if task == "task2b":
        dynamic = [row for row in rows if row.get("configuration") == "dynamic_irrip"
                   and row.get("threshold") not in (None, "")]
        thresholds = sorted({float(row["threshold"]) for row in dynamic})
        if thresholds:
            threshold_categories = [f"{value:.2f}" for value in thresholds]
            for metric in REPORT_METRICS[task]:
                series = []
                for trace in traces:
                    values_by_threshold = {}
                    for row in dynamic:
                        if row["trace"] != trace:
                            continue
                        value = get_number(row, metric)
                        if value is not None:
                            values_by_threshold[f"{float(row['threshold']):.2f}"] = value
                    if values_by_threshold:
                        series.append((trace_labels[traces.index(trace)], values_by_threshold))
                if not series:
                    continue
                fig_width = max(9.5, 1.5 * len(threshold_categories) + 1.7 * len(series))
                fig, ax = plt.subplots(figsize=(fig_width, 5.7))
                grouped_bars(ax, list(range(len(threshold_categories))), threshold_categories,
                             series, metric, threshold_colors, "Dynamic I-RRIP threshold",
                             label_values=False)
                title = f"Task 2B — Threshold Sensitivity\n{metric_title(metric)} by Decision Threshold"
                if metric == "speedup":
                    title += " (zoomed around 1.0; 1.0 = no change)"
                ax.set_title(title, fontsize=12, fontweight="semibold", pad=12)
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17),
                          ncol=min(3, len(series)), frameon=False, fontsize=8.5,
                          title="Benchmark trace")
                fig.tight_layout(rect=(0, 0.08, 1, 1))
                save_figure(fig, output_dir / f"sensitivity_{metric}")
                plotted.append(f"sensitivity_{metric}")
                plt.close(fig)

    # Extra diagnostic plots: percentage deltas are easier to compare than tiny raw IPC changes.
    if task == "task1b":
        base_cfg, comparators = "baseline", ["ip_stride", "spp"]
        effect_specs = [
            ("ipc_change_vs_baseline_pct", "IPC Change vs No-Prefetch Baseline",
             "IPC change vs baseline (%)", "ipc", "change"),
            ("l1d_mpki_reduction_vs_baseline_pct", "L1D Load-MPKI Reduction vs Baseline",
             "L1D load-MPKI reduction (%)", "l1d_load_mpki", "reduction"),
        ]
        for filename, title, ylabel, metric, effect in effect_specs:
            series = []
            for cfg in comparators:
                vals = {}
                for trace, pretty in zip(traces, trace_labels):
                    ref = value_by_config_trace(base_cfg, metric, trace)
                    value = value_by_config_trace(cfg, metric, trace)
                    if ref not in (None, 0) and value is not None:
                        vals[pretty] = ((value / ref - 1) * 100 if effect == "change"
                                        else (ref - value) / ref * 100)
                if vals:
                    series.append((config_labels[cfg], vals))
            plot_effect_chart(filename, title, ylabel, trace_labels, series,
                              [config_colors[cfg] for cfg in comparators])
            plotted.append(filename[:-4])

    elif task == "task2a":
        for filename, title, ylabel, metric, effect in [
            ("ipc_change_vs_rrip_pct", "IPC Change vs RRIP", "IPC change vs RRIP (%)", "ipc", "change"),
            ("l2i_mpki_reduction_vs_rrip_pct", "L2 Instruction MPKI Reduction vs RRIP",
             "L2 instruction load-MPKI reduction (%)", "l2i_load_mpki", "reduction"),
        ]:
            vals = {}
            for trace, pretty in zip(traces, trace_labels):
                ref = value_by_config_trace("rrip", metric, trace)
                value = value_by_config_trace("irrip", metric, trace)
                if ref not in (None, 0) and value is not None:
                    vals[pretty] = ((value / ref - 1) * 100 if effect == "change"
                                    else (ref - value) / ref * 100)
            plot_effect_chart(filename, title, ylabel, trace_labels,
                              [("I-RRIP vs RRIP", vals)], [config_colors["irrip"]])
            plotted.append(filename[:-4])

    elif task == "task2b":
        dynamic = [row for row in rows if row.get("configuration") == "dynamic_irrip"
                   and row.get("threshold") not in (None, "")]
        thresholds = sorted({float(row["threshold"]) for row in dynamic})
        threshold_categories = [f"{value:.2f}" for value in thresholds]
        for filename, title, ylabel, metric, effect in [
            ("ipc_change_vs_rrip_by_threshold_pct", "Dynamic I-RRIP IPC Change vs RRIP",
             "IPC change vs RRIP (%)", "ipc", "change"),
            ("l2i_mpki_reduction_vs_rrip_by_threshold_pct", "Dynamic I-RRIP Instruction-MPKI Reduction vs RRIP",
             "L2 instruction load-MPKI reduction (%)", "l2i_load_mpki", "reduction"),
        ]:
            series = []
            for trace in traces:
                vals = {}
                ref = value_by_config_trace("rrip", metric, trace)
                for row in dynamic:
                    if row["trace"] != trace:
                        continue
                    value = get_number(row, metric)
                    if ref not in (None, 0) and value is not None:
                        category = f"{float(row['threshold']):.2f}"
                        vals[category] = ((value / ref - 1) * 100 if effect == "change"
                                          else (ref - value) / ref * 100)
                if vals:
                    series.append((trace_labels[traces.index(trace)], vals))
            plot_effect_chart(filename, title, ylabel, threshold_categories, series,
                              threshold_colors, label_values=False)
            plotted.append(filename[:-4])

    elif task == "task3":
        # Assignment objective: verify whether combining SPP and Dynamic I-RRIP
        # improves IPC against the no-prefetch baseline and against the best standalone method.
        series = []
        for cfg in ("spp", "dynamic_irrip", "combined"):
            vals = {}
            for trace, pretty in zip(traces, trace_labels):
                ref = value_by_config_trace("baseline", "ipc", trace)
                value = value_by_config_trace(cfg, "ipc", trace)
                if ref not in (None, 0) and value is not None:
                    vals[pretty] = (value / ref - 1) * 100
            if vals:
                series.append((config_labels[cfg], vals))
        plot_effect_chart("ipc_change_vs_baseline_pct", "IPC Change vs No-Prefetch Baseline",
                          "IPC change vs baseline (%)", trace_labels, series,
                          [config_colors["spp"], config_colors["dynamic_irrip"], config_colors["combined"]])
        plotted.append("ipc_change_vs_baseline_pct")

        best_standalone_gains = {}
        for trace, pretty in zip(traces, trace_labels):
            spp = value_by_config_trace("spp", "ipc", trace)
            dynamic = value_by_config_trace("dynamic_irrip", "ipc", trace)
            combined = value_by_config_trace("combined", "ipc", trace)
            if spp is not None and dynamic is not None and combined is not None:
                best = max(spp, dynamic)
                if best != 0:
                    best_standalone_gains[pretty] = (combined / best - 1) * 100
        plot_effect_chart(
            "ipc_gain_vs_best_standalone_pct",
            "Combined Policy IPC Gain vs Best Standalone (SPP or Dynamic I-RRIP)",
            "Combined IPC change vs best standalone (%)", trace_labels,
            [("Combined vs best standalone", best_standalone_gains)], [config_colors["combined"]],
        )
        plotted.append("ipc_gain_vs_best_standalone_pct")

    print(f"Saved {len(plotted)} PNG plot(s) under {output_dir.relative_to(ROOT)}")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "run", "plot", "all", "reparse"))
    parser.add_argument("--task", choices=tuple(TASK_EXPERIMENTS), default="task1b")
    parser.add_argument("--stage", choices=tuple(EXPERIMENTS), default=None,
                        help="Run only one configuration in the selected task; omitted means run all stages")
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
                        help="Task 3 combined L2 policy (default: prefetch_irrip)")
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

    if args.stage is not None and args.command not in ("run", "all"):
        parser.error("--stage can only be used with the 'run' or 'all' command")
    if args.stage is not None and args.stage not in TASK_EXPERIMENTS[args.task]:
        parser.error(
            f"stage '{args.stage}' is not valid for {args.task}; "
            f"choose from: {', '.join(TASK_EXPERIMENTS[args.task])}"
        )

    plots_written = False
    if args.command in ("run", "all"):
        traces = find_traces(args.traces, args.traces_dir)
        thresholds = parse_thresholds(args.thresholds) if args.task == "task2b" else []
        rows = run_task(args.task, traces, args.warmup, args.simulation,
                        thresholds, args.fixed_threshold, args.debug,
                        resume=args.resume, stage=args.stage)
        if args.stage is not None:
            rows = merge_task_rows(args.task, rows)
        write_task_data(args.task, rows)
        # A targeted rerun refreshes charts from the merged, complete task dataset.
        if args.stage is not None:
            plot_task(args.task)
            plots_written = True
    if args.command in ("plot", "all") and not plots_written:
        plot_task(args.task)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
