#!/usr/bin/env python3
"""
Task 2C collector: software prefetching + SIMD.

This script never edits the assignment's source files. It generates patched
copies under .task2c_build/generated/ and benchmarks those copies.

Default staged experiment:
  1) no-SW-prefetch combined baselines for each SIMD width
  2) 256-bit combined-kernel prefetch-distance sweep (T0)
  3) choose best distance by geometric-mean speedup vs no-SW-prefetch
  4) locality-hint sweep at that distance
  5) choose best hint by geometric-mean speedup vs no-SW-prefetch
  6) SIMD-width sweep (128/256/512, supported widths only) at the selected
     distance/hint

Use --full-grid to run every supported width x distance x hint x size.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Tuple

COMMON_FLAGS = [
    "-std=c++17",
    "-O2",
    "-fno-tree-vectorize",
    "-Wall",
]

DEFAULT_SIZES = [128, 256, 512, 1024, 2048]
DEFAULT_DISTANCES = [16, 32, 64, 128, 256, 512]
DEFAULT_HINTS = ["T0", "T1", "T2", "NTA"]
DEFAULT_WIDTHS = [128, 256, 512]

WIDTH_CONFIG = {
    128: {
        "function": "task2c_alt_kernel",
        "required": ["avx", "fma"],
        "flags": ["-mavx", "-mfma", "-DTASK2C_BITS=128"],
        "source": "alt",
    },
    256: {
        "function": "matmul_optimized",
        "required": ["avx2", "fma"],
        "flags": ["-mavx2", "-mfma"],
        "source": "student",
    },
    512: {
        "function": "task2c_alt_kernel",
        "required": ["avx512f", "fma"],
        "flags": ["-mavx512f", "-mfma", "-DTASK2C_BITS=512"],
        "source": "alt",
    },
}

# Tried from left to right. The user's current Intel machine exposes
# sw_prefetch_access.any, so it is deliberately first.
EVENT_CANDIDATES = {
    "instructions": [
        "instructions",
        "cpu_core/instructions/",
    ],
    "cycles": [
        "cycles",
        "cpu_core/cycles/",
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
        "sw_prefetch_access.any",
        "cpu_core/sw_prefetch_access.any/",
        "SW_PREFETCH_ACCESS.ANY",
        "sw_prefetch_access",
    ],
}

CSV_FIELDS = [
    "host",
    "cpu",
    "hw_prefetch",
    "sweep",
    "stage",
    "width_bits",
    "size",
    "distance_floats",
    "distance_cache_lines",
    "hint",
    "sw_prefetch",
    "reps",
    "warmups",
    "perf_reps",
    "time_ms",
    "gflops",
    "speedup_vs_naive",
    "speedup_vs_no_sw_prefetch",
    "speedup_vs_simd_only_256",
    "naive_time_ms",
    "no_sw_prefetch_time_ms",
    "simd_only_256_time_ms",
    "instructions_per_kernel",
    "cycles_per_kernel",
    "ipc",
    "l1d_misses_per_kernel",
    "l2_misses_per_kernel",
    "llc_misses_per_kernel",
    "sw_prefetch_access_per_kernel",
    "perf_event_instructions",
    "perf_event_cycles",
    "perf_event_l1d_misses",
    "perf_event_l2_misses",
    "perf_event_llc_misses",
    "perf_event_sw_prefetch_access",
    "checksum",
]


def parse_int_list(text: str) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def parse_hint_list(text: str) -> List[str]:
    values = [x.strip().upper() for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    bad = [x for x in values if x not in DEFAULT_HINTS]
    if bad:
        raise argparse.ArgumentTypeError("unsupported hints: " + ", ".join(bad))
    return values


def run_cmd(cmd: List[str], check: bool = False):
    p = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and p.returncode != 0:
        print(p.stdout, end="")
        print(p.stderr, file=sys.stderr, end="")
        raise RuntimeError("command failed: " + " ".join(shlex.quote(x) for x in cmd))
    return p


def find_source(root: Path, candidates: Iterable[str]) -> Path:
    for rel in candidates:
        p = root / rel
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Could not find any of: " + ", ".join(candidates) + f" under {root}"
    )


def find_include_dir(root: Path, source: Path) -> Path:
    candidates = [root / "include", source.parent.parent / "include", source.parent, root]
    for d in candidates:
        if (d / "matmul.h").is_file():
            return d
    raise FileNotFoundError("Could not find matmul.h")


def cpu_flags() -> set[str]:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return set()
    for line in text.splitlines():
        if line.lower().startswith("flags"):
            return set(line.split(":", 1)[1].split())
    return set()


def supported_widths(requested: List[int]) -> Tuple[List[int], List[Tuple[int, str]]]:
    flags = cpu_flags()
    yes, no = [], []
    for width in dict.fromkeys(requested):
        if width not in WIDTH_CONFIG:
            raise ValueError(f"unsupported SIMD width: {width}")
        required = WIDTH_CONFIG[width]["required"]
        if not flags or all(x in flags for x in required):
            yes.append(width)
        else:
            no.append((width, "+".join(required)))
    return yes, no


def patch_prefetch(source_text: str, distance: int, hint: str) -> str:
    dpat = re.compile(r"(\b(?:const\s+)?int\s+prefetch_distance\s*=\s*)\d+(\s*;)")
    patched, n = dpat.subn(rf"\g<1>{distance}\g<2>", source_text, count=1)
    if n != 1:
        raise RuntimeError("Could not find exactly one prefetch_distance declaration")

    hpat = re.compile(r"_MM_HINT_(?:T0|T1|T2|NTA)")
    patched, n = hpat.subn(f"_MM_HINT_{hint}", patched)
    if n < 1:
        raise RuntimeError("Could not find any _MM_HINT_T0/T1/T2/NTA in kernel")
    return patched


def strip_software_prefetch(source_text: str) -> str:
    # Removes the whole conditional prefetch block, not just the intrinsic,
    # so the no-SW-prefetch baseline does not pay for a useless branch.
    pat = re.compile(
        r"\n\s*if\s*\(\s*p\s*\+\s*prefetch_distance\s*<\s*K\s*\)\s*\{\s*"
        r"(?:_mm_prefetch\([^;]+;\s*)+\}",
        re.S,
    )
    stripped, n = pat.subn("", source_text)
    if n < 1:
        raise RuntimeError(
            "Could not find the software-prefetch block. Original source was not changed."
        )
    return stripped


def compile_binary(
    cxx: str,
    driver: Path,
    kernel: Path,
    naive: Path,
    output: Path,
    function: str,
    include_dir: Path,
    extra_flags: List[str],
):
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        cxx,
        *COMMON_FLAGS,
        *extra_flags,
        f"-I{include_dir}",
        f"-DBENCH_FN={function}",
        str(driver),
        str(kernel),
    ]
    if kernel.resolve() != naive.resolve():
        cmd.append(str(naive))
    cmd += ["-o", str(output)]

    p = run_cmd(cmd)
    if p.returncode != 0:
        print(p.stdout, end="")
        print(p.stderr, file=sys.stderr, end="")
        raise RuntimeError("Compilation failed: " + " ".join(shlex.quote(x) for x in cmd))


def pin(cpu: Optional[int], cmd: List[str]) -> List[str]:
    if cpu is None:
        return cmd
    return ["taskset", "-c", str(cpu), *cmd]


def parse_json(stdout: str) -> dict:
    lines = [x.strip() for x in stdout.splitlines() if x.strip()]
    if not lines:
        raise RuntimeError("benchmark produced no JSON output")
    return json.loads(lines[-1])


def run_json(binary: Path, args: List[str], cpu: Optional[int], allow_fail=False):
    p = run_cmd(pin(cpu, [str(binary), *args]))
    if p.returncode != 0 and not allow_fail:
        print(p.stdout, end="")
        print(p.stderr, file=sys.stderr, end="")
        raise RuntimeError("benchmark failed")
    return parse_json(p.stdout), p.returncode


def perf_supported(event: str, cpu: Optional[int]) -> bool:
    cmd = ["perf", "stat", "-x", ";", "-e", event, "--", *pin(cpu, ["true"])]
    p = run_cmd(cmd)
    text = (p.stdout + "\n" + p.stderr).lower()
    bad = [
        "not supported",
        "event syntax error",
        "no permission",
        "permission denied",
        "access to performance monitoring",
    ]
    return p.returncode == 0 and not any(x in text for x in bad)


def resolve_perf_events(
    cpu: Optional[int], overrides: Dict[str, str], no_perf: bool
) -> Dict[str, Optional[str]]:
    if no_perf:
        return {k: None for k in EVENT_CANDIDATES}

    try:
        p = run_cmd(["perf", "--version"])
        if p.returncode != 0:
            raise FileNotFoundError
    except FileNotFoundError:
        print("warning: perf unavailable; collecting timing only", file=sys.stderr)
        return {k: None for k in EVENT_CANDIDATES}

    out: Dict[str, Optional[str]] = {}
    for logical, candidates in EVENT_CANDIDATES.items():
        choices = [overrides[logical]] if logical in overrides else candidates
        out[logical] = next((e for e in choices if perf_supported(e, cpu)), None)
    return out


def normalize_event(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def parse_number(s: str) -> Optional[float]:
    s = s.strip().replace(",", "")
    if not s or s.startswith("<"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_perf(stderr: str, events: Dict[str, Optional[str]]) -> Dict[str, Optional[float]]:
    out = {k: None for k in events}
    expected = {k: normalize_event(v) if v else "" for k, v in events.items()}

    for line in stderr.splitlines():
        parts = line.split(";")
        if len(parts) < 3:
            continue
        count = parse_number(parts[0])
        seen = normalize_event(parts[2])
        if count is None or not seen:
            continue
        for logical, want in expected.items():
            if want and (seen == want or seen in want or want in seen):
                out[logical] = count
                break
    return out


def perf_run(
    binary: Path,
    args: List[str],
    cpu: Optional[int],
    events: Dict[str, Optional[str]],
) -> Tuple[dict, Dict[str, Optional[float]]]:
    selected = [x for x in events.values() if x]
    program = pin(cpu, [str(binary), *args])

    if not selected:
        p = run_cmd(program)
        if p.returncode != 0:
            raise RuntimeError("benchmark failed")
        return parse_json(p.stdout), {k: None for k in events}

    cmd = ["perf", "stat", "-x", ";"]
    for e in selected:
        cmd += ["-e", e]
    cmd += ["--", *program]

    p = run_cmd(cmd)
    if p.returncode != 0:
        print(p.stdout, end="")
        print(p.stderr, file=sys.stderr, end="")
        raise RuntimeError("perf run failed: " + " ".join(shlex.quote(x) for x in cmd))
    return parse_json(p.stdout), parse_perf(p.stderr, events)


def corrected_per_kernel(
    measured: Dict[str, Optional[float]],
    setup: Dict[str, Optional[float]],
    reps: int,
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for key, val in measured.items():
        base = setup.get(key)
        if val is None:
            out[key] = None
        elif base is None:
            out[key] = val / reps
        else:
            out[key] = max(0.0, val - base) / reps
    return out


def adaptive_perf_reps(time_ms: float, target_ms: float, lo: int, hi: int) -> int:
    if time_ms <= 0:
        return lo
    return max(lo, min(hi, int(math.ceil(target_ms / time_ms))))


def geometric_mean(values: List[float]) -> float:
    vals = [x for x in values if x > 0]
    if not vals:
        return float("-inf")
    return math.exp(sum(math.log(x) for x in vals) / len(vals))


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    SCRIPT_DIR = Path(__file__).resolve().parent
    TASK2_ROOT = SCRIPT_DIR.parent    
    ap.add_argument(
        "--project-root",
        type=Path,
        default=TASK2_ROOT
    )
    ap.add_argument("--driver", type=Path, default=here / "task2c_driver.cpp")
    ap.add_argument("--alt-width-source", type=Path, default=here / "task2c_alt_widths.cpp")
    ap.add_argument("--sizes", type=parse_int_list, default=DEFAULT_SIZES)
    ap.add_argument("--distances", type=parse_int_list, default=DEFAULT_DISTANCES)
    ap.add_argument("--hints", type=parse_hint_list, default=DEFAULT_HINTS)
    ap.add_argument("--widths", type=parse_int_list, default=DEFAULT_WIDTHS)
    ap.add_argument("--distance-hint", choices=DEFAULT_HINTS, default="T0")
    ap.add_argument("--full-grid", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cpu", type=int, default=None)
    ap.add_argument("--hw-label", default="default")
    ap.add_argument("--perf-target-ms", type=float, default=750.0)
    ap.add_argument("--perf-min-reps", type=int, default=1)
    ap.add_argument("--perf-max-reps", type=int, default=1000)
    ap.add_argument("--build-dir", type=Path, default=Path(".task2c_build"))
    ap.add_argument(
        "--output",
        type=Path,
        default=TASK2_ROOT / "results" / "task2c_results.csv"
    )    
    ap.add_argument("--cxx", default="g++")
    ap.add_argument("--no-perf", action="store_true")
    ap.add_argument("--event", action="append", default=[], metavar="LOGICAL=PERF_EVENT")
    args = ap.parse_args()

    if args.reps <= 0 or args.warmups < 0:
        ap.error("--reps must be > 0 and --warmups >= 0")
    if 256 not in args.widths:
        ap.error("include 256 in --widths because the staged prefetch sweep uses your actual optimized kernel")
    if any(w not in WIDTH_CONFIG for w in args.widths):
        ap.error("--widths supports only 128,256,512")

    root = args.project_root.resolve()
    driver = args.driver.resolve()
    alt_source = args.alt_width_source.resolve()
    build = (root / args.build_dir).resolve() if not args.build_dir.is_absolute() else args.build_dir.resolve()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    generated = build / "generated"
    bindir = build / "bin"
    generated.mkdir(parents=True, exist_ok=True)
    bindir.mkdir(parents=True, exist_ok=True)

    naive_src = find_source(root, ["src/matmul_naive.cpp", "matmul_naive.cpp"])
    simd_src = find_source(root, ["src/matmul_simd.cpp", "matmul_simd.cpp"])
    optimized_src = find_source(root, ["src/matmul_optimized.cpp", "matmul_optimized.cpp"])
    include_dir = find_include_dir(root, optimized_src)

    widths, skipped = supported_widths(args.widths)
    if 256 not in widths:
        raise RuntimeError("This machine does not report AVX2/FMA support; cannot run the student's 256-bit optimized kernel")

    overrides: Dict[str, str] = {}
    for item in args.event:
        if "=" not in item:
            ap.error("--event must be LOGICAL=PERF_EVENT")
        logical, event = item.split("=", 1)
        logical, event = logical.strip(), event.strip()
        if logical not in EVENT_CANDIDATES:
            ap.error("unknown logical event: " + logical)
        overrides[logical] = event

    print("SIMD widths:")
    for w in args.widths:
        print(f"  {w:3d}-bit: {'YES' if w in widths else 'NO - skipped'}")

    print("\nResolving perf events:")
    events = resolve_perf_events(args.cpu, overrides, args.no_perf)
    for logical, event in events.items():
        print(f"  {logical:20s}: {event or 'UNAVAILABLE'}")

    # Compile invariant baselines.
    naive_bin = bindir / "naive"
    compile_binary(
        args.cxx, driver, naive_src, naive_src, naive_bin,
        "matmul_naive", include_dir, ["-mavx2", "-mfma"]
    )

    simd_bin = bindir / "simd_only_256"
    compile_binary(
        args.cxx, driver, simd_src, naive_src, simd_bin,
        "matmul_simd", include_dir, ["-mavx2", "-mfma"]
    )

    student_text = optimized_src.read_text()
    alt_text = alt_source.read_text()

    source_text_by_width = {
        128: alt_text,
        256: student_text,
        512: alt_text,
    }

    compiled: Dict[Tuple[str, int, int, str], Path] = {}

    def compile_variant(width: int, distance: int, hint: str, sw_prefetch: bool) -> Path:
        key = ("sw" if sw_prefetch else "nosw", width, distance, hint)
        if key in compiled:
            return compiled[key]

        cfg = WIDTH_CONFIG[width]
        text = source_text_by_width[width]
        if sw_prefetch:
            text = patch_prefetch(text, distance, hint)
        else:
            text = strip_software_prefetch(text)

        tag = f"w{width}_{'sw' if sw_prefetch else 'nosw'}"
        if sw_prefetch:
            tag += f"_d{distance}_{hint}"
        src = generated / f"{tag}.cpp"
        binary = bindir / tag
        src.write_text(text)

        compile_binary(
            args.cxx, driver, src, naive_src, binary,
            cfg["function"], include_dir, cfg["flags"]
        )
        compiled[key] = binary
        return binary

    # Representative correctness checks for every supported width.
    print("\nCorrectness checks at N=129:")
    for width in widths:
        b = compile_variant(width, 64, "T0", True)
        result, code = run_json(b, ["check", "129", str(args.seed)], args.cpu, allow_fail=True)
        ok = code == 0 and bool(result.get("correct", False))
        print(f"  {width:3d}-bit: {'correct' if ok else 'INCORRECT'}")
        if not ok:
            raise RuntimeError(f"{width}-bit combined kernel failed correctness")

    def collect_raw(binary: Path, size: int):
        timing, _ = run_json(
            binary,
            ["time", str(size), str(args.reps), str(args.warmups), str(args.seed)],
            args.cpu,
        )
        ms = float(timing["time_ms"])
        perf_reps = adaptive_perf_reps(
            ms, args.perf_target_ms, args.perf_min_reps, args.perf_max_reps
        )

        if any(events.values()):
            _, measured = perf_run(
                binary,
                ["perf", str(size), str(perf_reps), str(args.seed)],
                args.cpu,
                events,
            )
            _, setup = perf_run(
                binary,
                ["setup", str(size), str(args.seed)],
                args.cpu,
                events,
            )
            counters = corrected_per_kernel(measured, setup, perf_reps)
        else:
            counters = {k: None for k in events}

        return timing, perf_reps, counters

    rows: List[dict] = []
    naive_time: Dict[int, float] = {}
    simd_time: Dict[int, float] = {}
    no_sw_time: Dict[Tuple[int, int], float] = {}

    def make_row(
        *,
        sweep: str,
        stage: str,
        width: int,
        size: int,
        distance: int,
        hint: str,
        sw_prefetch: bool,
        timing: dict,
        perf_reps: int,
        counters: Dict[str, Optional[float]],
    ) -> dict:
        ntime = naive_time.get(size, float(timing["time_ms"]))
        swbase = no_sw_time.get((width, size))
        stime = simd_time.get(size)
        ms = float(timing["time_ms"])

        if stage == "naive":
            speed_naive = 1.0
        else:
            speed_naive = ntime / ms

        speed_no_sw = None
        if sw_prefetch and swbase is not None:
            speed_no_sw = swbase / ms

        speed_simd = None
        if width == 256 and stage != "simd_only" and stime is not None:
            speed_simd = stime / ms

        ins = counters.get("instructions")
        cyc = counters.get("cycles")

        row = {
            "host": socket.gethostname(),
            "cpu": "" if args.cpu is None else args.cpu,
            "hw_prefetch": args.hw_label,
            "sweep": sweep,
            "stage": stage,
            "width_bits": width,
            "size": size,
            "distance_floats": distance if sw_prefetch else 0,
            "distance_cache_lines": (distance / 16.0) if sw_prefetch else 0,
            "hint": hint if sw_prefetch else "NONE",
            "sw_prefetch": "on" if sw_prefetch else "off",
            "reps": args.reps,
            "warmups": args.warmups,
            "perf_reps": perf_reps,
            "time_ms": ms,
            "gflops": float(timing["gflops"]),
            "speedup_vs_naive": speed_naive,
            "speedup_vs_no_sw_prefetch": "" if speed_no_sw is None else speed_no_sw,
            "speedup_vs_simd_only_256": "" if speed_simd is None else speed_simd,
            "naive_time_ms": ntime,
            "no_sw_prefetch_time_ms": "" if swbase is None else swbase,
            "simd_only_256_time_ms": "" if stime is None else stime,
            "instructions_per_kernel": "" if ins is None else ins,
            "cycles_per_kernel": "" if cyc is None else cyc,
            "ipc": "" if safe_div(ins, cyc) is None else safe_div(ins, cyc),
            "l1d_misses_per_kernel": "" if counters.get("l1d_misses") is None else counters["l1d_misses"],
            "l2_misses_per_kernel": "" if counters.get("l2_misses") is None else counters["l2_misses"],
            "llc_misses_per_kernel": "" if counters.get("llc_misses") is None else counters["llc_misses"],
            "sw_prefetch_access_per_kernel": "" if counters.get("sw_prefetch_access") is None else counters["sw_prefetch_access"],
            "checksum": float(timing["checksum"]),
        }
        for logical in EVENT_CANDIDATES:
            row[f"perf_event_{logical}"] = events.get(logical) or ""
        return row

    print("\nBaselines:")
    for size in args.sizes:
        timing, pr, counters = collect_raw(naive_bin, size)
        naive_time[size] = float(timing["time_ms"])
        rows.append(make_row(
            sweep="baseline", stage="naive", width=0, size=size,
            distance=0, hint="NONE", sw_prefetch=False,
            timing=timing, perf_reps=pr, counters=counters,
        ))
        print(f"  N={size:4d} naive: {naive_time[size]:.3f} ms")

        timing, pr, counters = collect_raw(simd_bin, size)
        simd_time[size] = float(timing["time_ms"])
        rows.append(make_row(
            sweep="baseline", stage="simd_only", width=256, size=size,
            distance=0, hint="NONE", sw_prefetch=False,
            timing=timing, perf_reps=pr, counters=counters,
        ))
        print(f"         simd-only-256: {simd_time[size]:.3f} ms")

    print("\nCombined no-SW-prefetch baselines:")
    for width in widths:
        b = compile_variant(width, 64, "T0", False)
        for size in args.sizes:
            timing, pr, counters = collect_raw(b, size)
            no_sw_time[(width, size)] = float(timing["time_ms"])
            row = make_row(
                sweep="baseline", stage="combined_no_sw", width=width, size=size,
                distance=0, hint="NONE", sw_prefetch=False,
                timing=timing, perf_reps=pr, counters=counters,
            )
            rows.append(row)
            print(
                f"  N={size:4d} width={width:3d}: {row['time_ms']:.3f} ms  "
                f"{row['speedup_vs_naive']:.3f}x vs naive"
            )

    measured_rows: Dict[Tuple[int, int, str, int], dict] = {}

    def measure_combined(width: int, distance: int, hint: str, size: int, sweep: str) -> dict:
        key = (width, distance, hint, size)
        if key in measured_rows:
            row = dict(measured_rows[key])
            row["sweep"] = sweep
            rows.append(row)
            return row

        b = compile_variant(width, distance, hint, True)
        timing, pr, counters = collect_raw(b, size)
        row = make_row(
            sweep=sweep, stage="combined", width=width, size=size,
            distance=distance, hint=hint, sw_prefetch=True,
            timing=timing, perf_reps=pr, counters=counters,
        )
        measured_rows[key] = dict(row)
        rows.append(row)
        return row

    if args.full_grid:
        print("\nRunning full width x distance x hint x size grid...")
        for width in widths:
            for distance in args.distances:
                for hint in args.hints:
                    for size in args.sizes:
                        row = measure_combined(width, distance, hint, size, "full_grid")
                        print(
                            f"  N={size:4d} w={width:3d} d={distance:4d} {hint:3s}: "
                            f"{row['time_ms']:.3f} ms  "
                            f"{row['speedup_vs_naive']:.3f}x naive  "
                            f"{float(row['speedup_vs_no_sw_prefetch']):.3f}x no-SW"
                        )
    else:
        distance_rows: List[dict] = []
        print(f"\n256-bit distance sweep ({args.distance_hint})...")
        for distance in args.distances:
            for size in args.sizes:
                row = measure_combined(256, distance, args.distance_hint, size, "distance")
                distance_rows.append(row)
                print(
                    f"  N={size:4d} d={distance:4d}: {row['time_ms']:.3f} ms  "
                    f"{float(row['speedup_vs_no_sw_prefetch']):.3f}x vs no-SW"
                )

        distance_scores = {}
        for distance in args.distances:
            vals = [
                float(r["speedup_vs_no_sw_prefetch"])
                for r in distance_rows
                if int(r["distance_floats"]) == distance
            ]
            distance_scores[distance] = geometric_mean(vals)
        best_distance = max(distance_scores, key=distance_scores.get)
        print(f"\nSelected distance: {best_distance} floats ({best_distance/16:.1f} cache lines)")

        hint_rows: List[dict] = []
        print("\n256-bit locality-hint sweep...")
        for hint in args.hints:
            for size in args.sizes:
                row = measure_combined(256, best_distance, hint, size, "level")
                hint_rows.append(row)
                print(
                    f"  N={size:4d} {hint:3s}: {row['time_ms']:.3f} ms  "
                    f"{float(row['speedup_vs_no_sw_prefetch']):.3f}x vs no-SW"
                )

        hint_scores = {}
        for hint in args.hints:
            vals = [
                float(r["speedup_vs_no_sw_prefetch"])
                for r in hint_rows if r["hint"] == hint
            ]
            hint_scores[hint] = geometric_mean(vals)
        best_hint = max(hint_scores, key=hint_scores.get)
        print(f"\nSelected hint: {best_hint}")

        print("\nSIMD-width sweep at selected prefetch parameters...")
        for width in widths:
            for size in args.sizes:
                row = measure_combined(width, best_distance, best_hint, size, "width")
                print(
                    f"  N={size:4d} width={width:3d}: {row['time_ms']:.3f} ms  "
                    f"{row['speedup_vs_naive']:.3f}x vs naive"
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})

    print(f"\nWrote {len(rows)} rows to: {output}")
    if skipped:
        print("Skipped unsupported widths: " + ", ".join(f"{w} ({why})" for w, why in skipped))
    print("No assignment source file was modified.")


if __name__ == "__main__":
    main()
