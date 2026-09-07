#!/usr/bin/env python3
"""
Task 2A collector.

This script NEVER edits src/matmul_naive.cpp or src/matmul_prefetch.cpp.
For each prefetch configuration it:
  1. reads matmul_prefetch.cpp,
  2. writes a patched COPY under .task2a_build/generated/,
  3. compiles that copy with the assignment's pinned flags,
  4. runs the benchmark pinned to one CPU (optional),
  5. records execution time, speedup, and perf counters in CSV.

Default "staged" sweep:
  - baseline for every matrix size
  - distance sweep for every size with T0
  - choose best distance by geometric-mean speedup
  - locality-hint sweep for every size at that chosen distance

Use --full-grid for every size x distance x hint combination.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import statistics
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Tuple


PINNED_FLAGS = [
    "-std=c++17",
    "-O2",
    "-fno-tree-vectorize",
    "-mavx2",
    "-mfma",
    "-Wall",
]

DEFAULT_SIZES = [128, 256, 512, 1024, 2048]
DEFAULT_DISTANCES = [16, 32, 64, 128, 256, 512]
DEFAULT_HINTS = ["T0", "T1", "T2", "NTA"]

# Candidate perf names are tried from left to right.
# If your lab machine exposes a different name, use --event LOGICAL=PERF_EVENT.
EVENT_CANDIDATES = {
    "instructions": [
        "instructions",
        "cpu_core/instructions/",
    ],
    "l1d_misses": [
        "L1-dcache-load-misses",
        "l1-dcache-load-misses",
        "cpu_core/L1D.REPLACEMENT/",
        "cpu_core/l1d.replacement/",
    ],
    "l2_misses": [
        "l2_rqsts.miss",
        "L2_RQSTS.MISS",
        "cpu_core/L2_RQSTS.MISS/",
        "cpu_core/l2_rqsts.miss/",
    ],
    "llc_misses": [
        "LLC-load-misses",
        "llc-load-misses",
        "cpu_core/MEM_LOAD_RETIRED.L3_MISS/",
        "cpu_core/mem_load_retired.l3_miss/",
        "cache-misses",
    ],
    "sw_prefetch_access": [
        "sw_prefetch_access",
        "SW_PREFETCH_ACCESS",
        "cpu_core/SW_PREFETCH_ACCESS.ALL/",
        "cpu_core/sw_prefetch_access.all/",
    ],
}

CSV_FIELDS = [
    "host",
    "cpu",
    "hw_prefetch",
    "sweep",
    "stage",
    "size",
    "distance_floats",
    "distance_cache_lines",
    "hint",
    "reps",
    "warmups",
    "time_ms",
    "gflops",
    "speedup",
    "instructions_total",
    "instructions_per_call",
    "l1d_misses_total",
    "l1d_misses_per_call",
    "l2_misses_total",
    "l2_misses_per_call",
    "llc_misses_total",
    "llc_misses_per_call",
    "sw_prefetch_access_total",
    "sw_prefetch_access_per_call",
    "perf_event_instructions",
    "perf_event_l1d_misses",
    "perf_event_l2_misses",
    "perf_event_llc_misses",
    "perf_event_sw_prefetch_access",
    "checksum",
]


def parse_int_list(text: str) -> List[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def parse_str_list(text: str) -> List[str]:
    values = [x.strip().upper() for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    bad = [x for x in values if x not in DEFAULT_HINTS]
    if bad:
        raise argparse.ArgumentTypeError(f"unsupported hint(s): {', '.join(bad)}")
    return values


def run_cmd(cmd: List[str], cwd: Optional[Path] = None, check: bool = True):
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def find_source(root: Path, candidates: Iterable[str]) -> Path:
    for rel in candidates:
        p = root / rel
        if p.is_file():
            return p
    joined = ", ".join(candidates)
    raise FileNotFoundError(f"Could not find any of: {joined} under {root}")


def find_include_dir(root: Path, source: Path) -> Optional[Path]:
    candidates = [
        root / "include",
        source.parent.parent / "include",
        source.parent,
        root,
    ]
    for d in candidates:
        if (d / "matmul.h").is_file():
            return d
    return None


def patch_prefetch_copy(source_text: str, distance: int, hint: str) -> str:
    distance_pattern = re.compile(
        r"(\b(?:const\s+)?int\s+prefetch_distance\s*=\s*)\d+(\s*;)"
    )
    patched, n_distance = distance_pattern.subn(
        rf"\g<1>{distance}\g<2>", source_text, count=1
    )
    if n_distance != 1:
        raise RuntimeError(
            "Could not find exactly one 'int prefetch_distance = ...;' in "
            "matmul_prefetch.cpp. The original file was not changed."
        )

    hint_pattern = re.compile(r"_MM_HINT_(?:T0|T1|T2|NTA)")
    patched, n_hint = hint_pattern.subn(f"_MM_HINT_{hint}", patched)
    if n_hint < 1:
        raise RuntimeError(
            "Could not find an _MM_HINT_T0/T1/T2/NTA use in matmul_prefetch.cpp. "
            "The original file was not changed."
        )
    return patched


def compile_binary(
    cxx: str,
    driver: Path,
    kernel_src: Path,
    output: Path,
    fn_name: str,
    include_dir: Path,
):
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        cxx,
        *PINNED_FLAGS,
        f"-I{include_dir}",
        f"-DBENCH_FN={fn_name}",
        str(driver),
        str(kernel_src),
        "-o",
        str(output),
    ]
    proc = run_cmd(cmd, check=False)
    if proc.returncode != 0:
        print(proc.stdout, end="")
        print(proc.stderr, file=sys.stderr, end="")
        raise RuntimeError(f"Compilation failed: {' '.join(shlex.quote(x) for x in cmd)}")


def make_pinned_command(cpu: Optional[int], binary: Path, args: List[str]) -> List[str]:
    cmd = [str(binary), *args]
    if cpu is not None:
        cmd = ["taskset", "-c", str(cpu), *cmd]
    return cmd


def perf_supported(event: str, cpu: Optional[int]) -> bool:
    cmd = ["perf", "stat", "-x", ";", "-e", event, "--"]
    if cpu is not None:
        cmd += ["taskset", "-c", str(cpu), "true"]
    else:
        cmd += ["true"]
    proc = run_cmd(cmd, check=False)
    text = (proc.stdout + "\n" + proc.stderr).lower()
    if proc.returncode != 0:
        return False
    bad = (
        "<not supported>",
        "not supported",
        "event syntax error",
        "no permission",
        "permission denied",
        "access to performance monitoring",
    )
    return not any(x in text for x in bad)


def resolve_perf_events(
    cpu: Optional[int],
    overrides: Dict[str, str],
    no_perf: bool,
) -> Dict[str, Optional[str]]:
    if no_perf:
        return {k: None for k in EVENT_CANDIDATES}

    try:
        run_cmd(["perf", "--version"])
    except FileNotFoundError:
        print("warning: perf not found; collecting timing only.", file=sys.stderr)
        return {k: None for k in EVENT_CANDIDATES}

    resolved: Dict[str, Optional[str]] = {}
    for logical, candidates in EVENT_CANDIDATES.items():
        if logical in overrides:
            candidates = [overrides[logical]]

        chosen = None
        for event in candidates:
            if perf_supported(event, cpu):
                chosen = event
                break
        resolved[logical] = chosen

    return resolved


def parse_perf_number(text: str) -> Optional[float]:
    t = text.strip().replace(",", "")
    if not t or t.startswith("<"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def normalize_event_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def parse_perf_output(
    stderr: str,
    resolved: Dict[str, Optional[str]],
) -> Dict[str, Optional[float]]:
    out = {k: None for k in resolved}
    normalized = {
        k: normalize_event_name(v) if v else ""
        for k, v in resolved.items()
    }

    for line in stderr.splitlines():
        parts = line.split(";")
        if len(parts) < 3:
            continue
        count = parse_perf_number(parts[0])
        event_seen = normalize_event_name(parts[2])
        if count is None or not event_seen:
            continue

        for logical, expected in normalized.items():
            if not expected:
                continue
            if event_seen == expected or event_seen in expected or expected in event_seen:
                out[logical] = count
                break
    return out


def run_benchmark(
    binary: Path,
    size: int,
    reps: int,
    warmups: int,
    seed: int,
    cpu: Optional[int],
    perf_events: Dict[str, Optional[str]],
) -> Tuple[dict, Dict[str, Optional[float]]]:

    program = make_pinned_command(
        cpu, binary, [str(size), str(reps), str(warmups), str(seed)]
    )
    event_list = [e for e in perf_events.values() if e]

    if event_list:
        cmd = ["perf", "stat", "-x", ";"]
        for event in event_list:
            cmd += ["-e", event]
        cmd += ["--", *program]
        proc = run_cmd(cmd, check=False)
    else:
        proc = run_cmd(program, check=False)

    if proc.returncode != 0:
        print(proc.stdout, end="")
        print(proc.stderr, file=sys.stderr, end="")
        raise RuntimeError(
            f"Benchmark failed: {' '.join(shlex.quote(x) for x in program)}"
        )

    lines = [x.strip() for x in proc.stdout.splitlines() if x.strip()]
    if not lines:
        raise RuntimeError("Benchmark produced no JSON output")
    try:
        bench = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse benchmark output: {lines[-1]}") from exc

    counters = (
        parse_perf_output(proc.stderr, perf_events)
        if event_list
        else {k: None for k in perf_events}
    )
    return bench, counters


def row_from_run(
    *,
    stage: str,
    sweep: str,
    size: int,
    distance: int,
    hint: str,
    reps: int,
    warmups: int,
    hw_label: str,
    cpu: Optional[int],
    baseline_ms: float,
    bench: dict,
    counters: Dict[str, Optional[float]],
    perf_events: Dict[str, Optional[str]],
) -> dict:
    calls = reps + warmups
    speedup = 1.0 if stage == "naive" else baseline_ms / float(bench["time_ms"])

    row = {
        "host": socket.gethostname(),
        "cpu": "" if cpu is None else cpu,
        "hw_prefetch": hw_label,
        "sweep": sweep,
        "stage": stage,
        "size": size,
        "distance_floats": distance if stage == "prefetch" else 0,
        "distance_cache_lines": (distance / 16.0) if stage == "prefetch" else 0,
        "hint": hint if stage == "prefetch" else "NONE",
        "reps": reps,
        "warmups": warmups,
        "time_ms": float(bench["time_ms"]),
        "gflops": float(bench["gflops"]),
        "speedup": speedup,
        "checksum": float(bench["checksum"]),
    }

    for logical in EVENT_CANDIDATES:
        total = counters.get(logical)
        row[f"{logical}_total"] = "" if total is None else total
        row[f"{logical}_per_call"] = (
            "" if total is None else total / max(calls, 1)
        )
        row[f"perf_event_{logical}"] = perf_events.get(logical) or ""

    return row


def geometric_mean(values: List[float]) -> float:
    vals = [v for v in values if v > 0]
    if not vals:
        return float("-inf")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def write_rows(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def main():
    parser = argparse.ArgumentParser()
    SCRIPT_DIR = Path(__file__).resolve().parent
    TASK2_ROOT = SCRIPT_DIR.parent
    parser.add_argument(
        "--project-root",
        type=Path,
        default=TASK2_ROOT
    )
    parser.add_argument("--driver", type=Path, default=Path(__file__).with_name("task2a_driver.cpp"))
    parser.add_argument(
        "--output",
        type=Path,
        default=TASK2_ROOT / "results" / "task2a_results.csv"
    )    
    parser.add_argument("--build-dir", type=Path, default=Path(".task2a_build"))
    parser.add_argument("--sizes", type=parse_int_list, default=DEFAULT_SIZES)
    parser.add_argument("--distances", type=parse_int_list, default=DEFAULT_DISTANCES)
    parser.add_argument("--hints", type=parse_str_list, default=DEFAULT_HINTS)
    parser.add_argument("--distance-hint", choices=DEFAULT_HINTS, default="T0")
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--hw-label", default="default")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--no-perf", action="store_true")
    parser.add_argument(
        "--event",
        action="append",
        default=[],
        metavar="LOGICAL=PERF_EVENT",
        help="Override a perf event. LOGICAL is one of: "
             + ", ".join(EVENT_CANDIDATES.keys()),
    )
    args = parser.parse_args()

    if args.reps <= 0 or args.warmups < 0:
        parser.error("--reps must be > 0 and --warmups must be >= 0")

    root = args.project_root.resolve()
    driver = args.driver.resolve()
    build = (root / args.build_dir).resolve() if not args.build_dir.is_absolute() else args.build_dir
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output

    naive_src = find_source(root, ["src/matmul_naive.cpp", "matmul_naive.cpp"])
    prefetch_src = find_source(root, ["src/matmul_prefetch.cpp", "matmul_prefetch.cpp"])
    include_dir = find_include_dir(root, prefetch_src)
    if include_dir is None:
        raise FileNotFoundError(
            "Could not find matmul.h. Run this from the Task-2 repository root "
            "or pass --project-root."
        )

    overrides: Dict[str, str] = {}
    for item in args.event:
        if "=" not in item:
            parser.error(f"bad --event value: {item}")
        logical, event = item.split("=", 1)
        logical = logical.strip()
        event = event.strip()
        if logical not in EVENT_CANDIDATES:
            parser.error(f"unknown logical event: {logical}")
        overrides[logical] = event

    print("Resolving perf events...")
    perf_events = resolve_perf_events(args.cpu, overrides, args.no_perf)
    for logical, event in perf_events.items():
        print(f"  {logical:20s}: {event or 'UNAVAILABLE'}")

    generated_dir = build / "generated"
    bin_dir = build / "bin"
    generated_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    naive_bin = bin_dir / "naive"
    print("\nCompiling naive benchmark...")
    compile_binary(
        args.cxx, driver, naive_src, naive_bin,
        "matmul_naive", include_dir
    )

    original_prefetch = prefetch_src.read_text()

    binaries: Dict[Tuple[int, str], Path] = {}

    def get_prefetch_binary(distance: int, hint: str) -> Path:
        key = (distance, hint)
        if key in binaries:
            return binaries[key]

        generated = generated_dir / f"matmul_prefetch_d{distance}_{hint}.cpp"
        generated.write_text(patch_prefetch_copy(original_prefetch, distance, hint))
        binary = bin_dir / f"prefetch_d{distance}_{hint}"
        compile_binary(
            args.cxx, driver, generated, binary,
            "matmul_prefetch", include_dir
        )
        binaries[key] = binary
        return binary

    rows: List[dict] = []
    baselines: Dict[int, float] = {}

    print("\nRunning baselines...")
    for size in args.sizes:
        bench, counters = run_benchmark(
            naive_bin, size, args.reps, args.warmups, args.seed,
            args.cpu, perf_events
        )
        baselines[size] = float(bench["time_ms"])
        row = row_from_run(
            stage="naive", sweep="baseline", size=size, distance=0, hint="NONE",
            reps=args.reps, warmups=args.warmups, hw_label=args.hw_label,
            cpu=args.cpu, baseline_ms=baselines[size], bench=bench,
            counters=counters, perf_events=perf_events,
        )
        rows.append(row)
        print(f"  N={size:4d}: {row['time_ms']:.3f} ms")

    if args.full_grid:
        print("\nRunning full distance x hint grid...")
        for distance in args.distances:
            for hint in args.hints:
                binary = get_prefetch_binary(distance, hint)
                for size in args.sizes:
                    bench, counters = run_benchmark(
                        binary, size, args.reps, args.warmups, args.seed,
                        args.cpu, perf_events
                    )
                    row = row_from_run(
                        stage="prefetch", sweep="full_grid",
                        size=size, distance=distance, hint=hint,
                        reps=args.reps, warmups=args.warmups,
                        hw_label=args.hw_label, cpu=args.cpu,
                        baseline_ms=baselines[size], bench=bench,
                        counters=counters, perf_events=perf_events,
                    )
                    rows.append(row)
                    print(
                        f"  N={size:4d} d={distance:4d} {hint:3s}: "
                        f"{row['time_ms']:.3f} ms  {row['speedup']:.3f}x"
                    )
    else:
        distance_rows: List[dict] = []

        print(f"\nDistance sweep using {args.distance_hint}...")
        for distance in args.distances:
            binary = get_prefetch_binary(distance, args.distance_hint)
            for size in args.sizes:
                bench, counters = run_benchmark(
                    binary, size, args.reps, args.warmups, args.seed,
                    args.cpu, perf_events
                )
                row = row_from_run(
                    stage="prefetch", sweep="distance",
                    size=size, distance=distance, hint=args.distance_hint,
                    reps=args.reps, warmups=args.warmups,
                    hw_label=args.hw_label, cpu=args.cpu,
                    baseline_ms=baselines[size], bench=bench,
                    counters=counters, perf_events=perf_events,
                )
                rows.append(row)
                distance_rows.append(row)
                print(
                    f"  N={size:4d} d={distance:4d}: "
                    f"{row['time_ms']:.3f} ms  {row['speedup']:.3f}x"
                )

        scores = {}
        for distance in args.distances:
            vals = [
                float(r["speedup"])
                for r in distance_rows
                if int(r["distance_floats"]) == distance
            ]
            scores[distance] = geometric_mean(vals)
        best_distance = max(scores, key=scores.get)
        print(
            f"\nSelected distance for locality-hint sweep: "
            f"{best_distance} floats ({best_distance / 16:.1f} cache lines)"
        )

        print("\nLocality-hint sweep...")
        for hint in args.hints:
            binary = get_prefetch_binary(best_distance, hint)
            for size in args.sizes:
                # Avoid rerunning the one configuration already present in distance sweep.
                if hint == args.distance_hint:
                    existing = next(
                        r for r in distance_rows
                        if int(r["size"]) == size
                        and int(r["distance_floats"]) == best_distance
                        and r["hint"] == hint
                    )
                    level_row = dict(existing)
                    level_row["sweep"] = "level"
                    rows.append(level_row)
                    print(
                        f"  N={size:4d} {hint:3s}: "
                        f"{level_row['time_ms']:.3f} ms  "
                        f"{level_row['speedup']:.3f}x (reused)"
                    )
                    continue

                bench, counters = run_benchmark(
                    binary, size, args.reps, args.warmups, args.seed,
                    args.cpu, perf_events
                )
                row = row_from_run(
                    stage="prefetch", sweep="level",
                    size=size, distance=best_distance, hint=hint,
                    reps=args.reps, warmups=args.warmups,
                    hw_label=args.hw_label, cpu=args.cpu,
                    baseline_ms=baselines[size], bench=bench,
                    counters=counters, perf_events=perf_events,
                )
                rows.append(row)
                print(
                    f"  N={size:4d} {hint:3s}: "
                    f"{row['time_ms']:.3f} ms  {row['speedup']:.3f}x"
                )

    write_rows(output, rows)
    print(f"\nWrote {len(rows)} rows to: {output}")
    print("Original assignment source files were not modified.")


if __name__ == "__main__":
    main()
