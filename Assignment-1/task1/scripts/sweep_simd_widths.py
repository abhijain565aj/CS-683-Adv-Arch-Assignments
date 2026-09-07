#!/usr/bin/env python3
"""Run in task1. Build isolated SIMD-width variants and save measurements."""
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


def run(command, cwd, log):
    result = subprocess.run(list(map(str, command)), cwd=cwd,
                            env=dict(os.environ, LC_ALL='C'), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text('$ ' + shlex.join(list(map(str, command))) + '\n' + result.stdout)
    if result.returncode:
        raise RuntimeError(f'Command failed; see {log}\n{result.stdout[-2500:]}')
    return result.stdout


def cpu_flags(cpu):
    for block in Path('/proc/cpuinfo').read_text().split('\n\n'):
        fields = dict(line.split(':', 1) for line in block.splitlines() if ':' in line)
        fields = {k.strip(): v.strip() for k, v in fields.items()}
        if fields.get('processor') == str(cpu):
            return set(fields.get('flags', '').split())
    raise RuntimeError(f'CPU {cpu} was not found in /proc/cpuinfo')


def variant(source, width):
    # The substitutions match the uploaded kernel; only temporary copies change.
    if 'ox += 8' not in source or '_mm256_fmadd_ps' not in source:
        raise RuntimeError('Expected the uploaded 8-column AVX2 conv_simd.cpp.')
    source = source.replace('ox += 8', f'ox += {width // 32}')
    if width == 128:
        source = source.replace('__m256', '__m128').replace('_mm256_', '_mm_')
    elif width == 512:
        source = source.replace('__m256', '__m512').replace('_mm256_', '_mm512_')
    return source


def timings(text):
    values = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] not in ('naive', 'simd'):
            continue
        if len(fields) > 1 and fields[1] == '(ref)':
            fields.pop(1)
        if len(fields) < 5 or fields[1] not in ('yes', 'no'):
            continue
        if fields[1] == 'no':
            raise RuntimeError('Correctness check failed: ' + line)
        values[fields[0]] = float(fields[2])
    if set(values) != {'naive', 'simd'} or min(values.values()) <= 0:
        raise RuntimeError('Could not parse positive naive and SIMD timings.')
    return values


def instruction_count(text):
    for line in text.splitlines():
        fields = line.split(';')
        if len(fields) >= 3 and '/instructions/' in fields[2]:
            try:
                value = float(fields[0].strip())
            except ValueError:
                raise RuntimeError('Instructions counter unavailable: ' + line)
            if math.isfinite(value) and value > 0:
                return value
    raise RuntimeError('Missing valid instruction count; inspect the perf log.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    SCRIPT_DIR = Path(__file__).resolve().parent
    TASK1_ROOT = SCRIPT_DIR.parent
    parser.add_argument(
        '--root',
        type=Path,
        default=TASK1_ROOT
    )
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--kernel', type=int, default=3)
    parser.add_argument('--cpu', type=int, default=0, help='P-core logical CPU')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--perf-repeats', type=int, default=1)
    parser.add_argument('--timing-repeats', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.kernel <= 0 or args.kernel % 2 == 0:
        parser.error('kernel must be positive and odd')
    if min(args.perf_repeats, args.timing_repeats) < 1:
        parser.error('repeat counts must be positive')
    root = args.root.resolve()
    for relative in ('src/conv_simd.cpp', 'src/main.cpp', 'perf_main.cpp', 'include'):
        if not (root / relative).exists():
            raise RuntimeError(f'Missing {root / relative}; run from task1 or set --root.')
    flags = cpu_flags(args.cpu)
    if not {'avx2', 'fma'} <= flags:
        raise RuntimeError('The selected CPU must support AVX2 and FMA for the assignment build.')
    widths = [128, 256]
    if 'avx512f' in flags:
        widths.append(512)
    else:
        print('512-bit: skipped because AVX-512F is unavailable on the selected CPU.')
    original = (root / 'src/conv_simd.cpp').read_text()
    variant(original, 128)
    output = (
        args.output
        or root / 'results' / ('simd_results_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    logs = output / 'logs'
    logs.mkdir()
    paths = [p for folder in ('src', 'include') for p in (root / folder).rglob('*') if p.is_file()]
    paths.append(root / 'perf_main.cpp')
    metadata = dict(kernel=args.kernel, cpu=args.cpu, seed=args.seed,
                    widths=widths, skipped_widths=[w for w in [128,256,512] if w not in widths],
                    sizes=SIZES, flags=FLAGS, perf_repeats=args.perf_repeats,
                    timing_repeats=args.timing_repeats,
                    source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    counter_scope='Whole perf_main process, including setup, 5 warmups and 20 kernel calls',
                    speedup='Median of paired naive_ms/simd_ms timing ratios',
                    implementation='128/256/512-bit FMA; 128-bit uses AVX-encoded instructions, not legacy SSE-only flags')
    (output / 'settings.json').write_text(json.dumps(metadata, indent=2))
    columns = ['H', 'W', 'K', 'simd_width', 'naive_instructions', 'simd_instructions',
               'instruction_reduction', 'naive_time_ms', 'simd_time_ms', 'speedup',
               'cpu', 'perf_repeats', 'timing_repeats']
    compiler = shlex.split(os.environ.get('CXX', 'g++'))
    print(f'Sizes: {SIZES}; widths: {widths}; K={args.kernel}\nOutput: {output}', flush=True)
    with tempfile.TemporaryDirectory(prefix='conv_simd_sweep_') as directory:
        work = Path(directory)
        shutil.copytree(root / 'src', work / 'src')
        shutil.copytree(root / 'include', work / 'include')
        shutil.copy2(root / 'perf_main.cpp', work / 'perf_main.cpp')
        base = compiler + FLAGS + ['-I' + str(work / 'include')]
        objects = []
        # Compile the baseline and other stages once with the pinned flags.
        for src in sorted((work / 'src').glob('conv_*.cpp')):
            if src.name == 'conv_simd.cpp':
                continue
            obj = work / (src.stem + '.o')
            run(base + ['-c', src, '-o', obj], work, logs / (src.stem + '_build.log'))
            objects.append(obj)
        with (output / 'simd_width_results.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            stream.flush()
            for width in widths:
                print(f'\nBuilding {width}-bit variant', flush=True)
                src = work / 'src/conv_simd.cpp'
                src.write_text(variant(original, width))
                obj = work / f'simd_{width}.o'
                extra = ['-mavx512f'] if width == 512 else []
                run(base + extra + ['-c', src, '-o', obj], work, logs / f'build_simd_{width}.log')
                timing_bin, perf_bin = work / f'conv_{width}', work / f'perf_{width}'
                for main_src, binary, label in [(work / 'src/main.cpp', timing_bin, 'timing'),
                                                (work / 'perf_main.cpp', perf_bin, 'perf')]:
                    run(base + [main_src] + objects + [obj, '-o', binary], work,
                        logs / f'build_{label}_{width}.log')
                for n in SIZES:
                    print(f'\nH=W={n}, width={width}, K={args.kernel}', flush=True)
                    prefix = ['taskset', '-c', str(args.cpu)]
                    dimensions = list(map(str, [n, n, args.kernel, args.seed]))
                    samples = []
                    for repeat in range(args.timing_repeats):
                        print(f'  timing {repeat+1}/{args.timing_repeats}', flush=True)
                        text = run(prefix + [timing_bin, 'simd'] + dimensions, work,
                                   logs / f'n{n}_w{width}_time{repeat}.log')
                        samples.append(timings(text))
                    counts = {}
                    for stage in ('naive', 'simd'):
                        print(f'  {stage} instructions', flush=True)
                        text = run(prefix + ['perf', 'stat', '--no-big-num', '-x', ';',
                                   '-r', str(args.perf_repeats), '-e', 'cpu_core/instructions/',
                                   perf_bin, stage] + dimensions, work,
                                   logs / f'n{n}_w{width}_{stage}_perf.log')
                        counts[stage] = instruction_count(text)
                    row = dict(H=n, W=n, K=args.kernel, simd_width=width,
                               naive_instructions=counts['naive'], simd_instructions=counts['simd'],
                               instruction_reduction=counts['naive']/counts['simd'],
                               naive_time_ms=statistics.median(s['naive'] for s in samples),
                               simd_time_ms=statistics.median(s['simd'] for s in samples),
                               speedup=statistics.median(s['naive']/s['simd'] for s in samples),
                               cpu=args.cpu, perf_repeats=args.perf_repeats, timing_repeats=args.timing_repeats)
                    writer.writerow(row)
                    stream.flush()
                    print(f"  SIMD instructions={counts['simd']:.0f}; speedup={row['speedup']:.3f}x", flush=True)
    print(f"\nSaved: {output / 'simd_width_results.csv'}")


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError) as error:
        raise SystemExit(str(error))
