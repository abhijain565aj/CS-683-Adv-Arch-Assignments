#!/usr/bin/env python3
"""Run from task1. Original sources and executables are never modified."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import tempfile
from datetime import datetime

SIZES = [512, 1024, 2048, 4096, 8192]
FLAGS = ['-std=c++17', '-O2', '-fno-tree-vectorize', '-mavx2', '-mfma', '-Wall']


def tile_sizes(n):
    return [n // (2 ** power) for power in range(6, 0, -1)]


def run(command, cwd, log):
    env = dict(os.environ, LC_ALL='C')
    result = subprocess.run(list(map(str, command)), cwd=cwd, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log.write_text('$ ' + shlex.join(list(map(str, command))) + '\n' + result.stdout)
    if result.returncode:
        raise RuntimeError(f'Command failed; see {log}\n{result.stdout[-3000:]}')
    return result.stdout


def parse_timing(text):
    rows = {}
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] in ('naive', 'tile'):
            if len(fields) > 1 and fields[1] == '(ref)':
                fields.pop(1)
            if len(fields) < 5 or fields[1] not in ('yes', 'no'):
                continue
            if fields[1] != 'yes':
                raise RuntimeError('Correctness failed: ' + line)
            rows[fields[0]] = float(fields[2])
    if set(rows) != {'naive', 'tile'} or min(rows.values()) <= 0:
        raise RuntimeError('Cannot read positive naive/tile timings from harness output.')
    return rows


def parse_counters(text):
    values = {}
    for line in text.splitlines():
        fields = line.split(';')
        if len(fields) < 3:
            continue
        event = fields[2].strip()
        for name, suffix in [('instructions', '/instructions/'),
                             ('l1_misses', '/mem_load_retired.l1_miss/')]:
            if suffix in event:
                try:
                    value = float(fields[0].strip())
                except ValueError:
                    raise RuntimeError(f'Counter unavailable: {line}')
                if not math.isfinite(value) or value < 0:
                    raise RuntimeError(f'Invalid counter: {line}')
                values[name] = value
    if set(values) != {'instructions', 'l1_misses'} or values['instructions'] <= 0:
        raise RuntimeError('Could not read both counters; inspect the saved perf log.')
    return values


def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    TASK1_ROOT = SCRIPT_DIR.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--root',
        type=Path,
        default=TASK1_ROOT,
        help='task1 directory'
    )
    parser.add_argument('--kernel', type=int, default=3)
    parser.add_argument('--cpu', type=int, default=0, help='P-core logical CPU')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--perf-repeats', type=int, default=1)
    parser.add_argument('--timing-repeats', type=int, default=3)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan', action='store_true', help='Print pairs without building or running')
    args = parser.parse_args()
    if args.kernel <= 0 or args.kernel % 2 == 0:
        parser.error('kernel must be positive and odd')
    if min(args.perf_repeats, args.timing_repeats) < 1:
        parser.error('repeat counts must be positive')
    for n in SIZES:
        print(f'{n:5d} x {n:<5d}: {tile_sizes(n)}', flush=True)
    if args.plan:
        return

    root = args.root.resolve()
    required = [root / 'src', root / 'include', root / 'perf_main.cpp',
                root / 'src/main.cpp', root / 'src/conv_tile.cpp']
    for path in required:
        if not path.exists():
            raise RuntimeError(f'Missing {path}; use --root with your task1 directory.')
    for command in ('perf', 'taskset'):
        if not shutil.which(command):
            raise RuntimeError(f'{command} not found')
    original = (root / 'src/conv_tile.cpp').read_text()
    pattern = r'\bint\s+sz\s*=\s*\d+\s*;'
    if len(re.findall(pattern, original)) != 1:
        raise RuntimeError('Expected exactly one declaration such as int sz = 32;')

    output = (
        args.output
        or root / 'results' / ('tile_results_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    logs = output / 'logs'
    logs.mkdir()
    source_paths = [p for folder in ('src', 'include') for p in (root / folder).rglob('*') if p.is_file()]
    source_paths.append(root / 'perf_main.cpp')
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    metadata = dict(kernel=args.kernel, cpu=args.cpu, seed=args.seed,
                    perf_repeats=args.perf_repeats, timing_repeats=args.timing_repeats,
                    flags=FLAGS, source_sha256=hashes,
                    counter_scope='Entire original perf_main process, including setup and all kernel calls',
                    speedup='Median of paired naive_ms/tile_ms ratios from assignment harness',
                    tile_rule='N/64, N/32, N/16, N/8, N/4, N/2')
    (output / 'settings.json').write_text(json.dumps(metadata, indent=2))
    fields = ['H', 'W', 'K', 'tile_size', 'instructions', 'l1_misses', 'l1_mpki',
              'naive_time_ms', 'tile_time_ms', 'speedup', 'perf_repeats', 'timing_repeats', 'cpu']
    results = []
    compiler = shlex.split(os.environ.get('CXX', 'g++'))
    print(f'K={args.kernel}; output: {output}', flush=True)
    with tempfile.TemporaryDirectory(prefix='conv_tile_sweep_') as temp:
        work = Path(temp)
        shutil.copytree(root / 'src', work / 'src')
        shutil.copytree(root / 'include', work / 'include')
        shutil.copy2(root / 'perf_main.cpp', work / 'perf_main.cpp')
        kernels = [str(p) for p in sorted((work / 'src').glob('conv_*.cpp'))]
        built = {}
        with (output / 'tile_results.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            stream.flush()
            for n in SIZES:
                for tile in tile_sizes(n):
                    print(f'\nH=W={n}, tile={tile}, K={args.kernel}', flush=True)
                    if tile not in built:
                        (work / 'src/conv_tile.cpp').write_text(re.sub(pattern, f'int sz = {tile};', original))
                        timing_bin = work / f'conv_{tile}'
                        perf_bin = work / f'perf_{tile}'
                        base = compiler + FLAGS + ['-I' + str(work / 'include')] + kernels
                        run(base + [work / 'src/main.cpp', '-o', timing_bin], work, logs / f'build_timing_{tile}.log')
                        run(base + [work / 'perf_main.cpp', '-o', perf_bin], work, logs / f'build_perf_{tile}.log')
                        built[tile] = (timing_bin, perf_bin)
                    timing_bin, perf_bin = built[tile]
                    prefix = ['taskset', '-c', str(args.cpu)]
                    dimensions = [str(n), str(n), str(args.kernel), str(args.seed)]
                    samples = []
                    for repeat in range(args.timing_repeats):
                        print(f'  timing {repeat + 1}/{args.timing_repeats}', flush=True)
                        text = run(prefix + [timing_bin, 'tile'] + dimensions, work,
                                   logs / f'n{n}_tile{tile}_time{repeat}.log')
                        samples.append(parse_timing(text))
                    print(f'  counters ({args.perf_repeats} process repetitions)', flush=True)
                    events = '{cpu_core/instructions/,cpu_core/mem_load_retired.l1_miss/}'
                    text = run(prefix + ['perf', 'stat', '--no-big-num', '-x', ';',
                               '-r', str(args.perf_repeats), '-e', events,
                               perf_bin, 'tile'] + dimensions, work, logs / f'n{n}_tile{tile}_perf.log')
                    counters = parse_counters(text)
                    row = dict(H=n, W=n, K=args.kernel, tile_size=tile, **counters,
                               l1_mpki=1000 * counters['l1_misses'] / counters['instructions'],
                               naive_time_ms=statistics.median(s['naive'] for s in samples),
                               tile_time_ms=statistics.median(s['tile'] for s in samples),
                               speedup=statistics.median(s['naive']/s['tile'] for s in samples),
                               perf_repeats=args.perf_repeats, timing_repeats=args.timing_repeats, cpu=args.cpu)
                    writer.writerow(row)
                    stream.flush()
                    results.append(row)
                    print(f"  instructions={row['instructions']:.0f}  L1-D MPKI={row['l1_mpki']:.6f}  speedup={row['speedup']:.3f}x", flush=True)
    print('\nBest measured tile sizes:')
    for n in SIZES:
        subset = [r for r in results if r['H'] == n]
        fastest = min(subset, key=lambda r: r['tile_time_ms'])
        lowest = min(subset, key=lambda r: r['l1_mpki'])
        print(f"{n}: lowest time tile={fastest['tile_size']}; lowest MPKI tile={lowest['tile_size']}")
    print(f"Saved: {output / 'tile_results.csv'}")


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError) as error:
        raise SystemExit(str(error))
