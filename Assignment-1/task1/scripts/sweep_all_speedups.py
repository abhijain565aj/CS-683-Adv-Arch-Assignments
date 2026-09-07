#!/usr/bin/env python3
"""Run all convolution stages and produce two grouped speedup bar charts."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

STAGES = ['naive', 'unroll', 'reorder', 'tile', 'simd', 'optimized']
LABELS = ['Naive', 'Loop unrolling', 'Loop reordering', 'Tiling', 'SIMD', 'Optimized']
COLORS = ['#777777', '#4472c4', '#ed7d31', '#a5a5a5', '#ffc000', '#5b9bd5']
SIZES = [512, 1024, 2048, 4096, 8192]
KERNELS = [3, 5, 7, 9]


def parse_table(text):
    times = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] not in STAGES:
            continue
        if len(fields) > 1 and fields[1] == '(ref)':
            fields.pop(1)
        if len(fields) < 5 or fields[1] not in ('yes', 'no'):
            continue
        if fields[1] != 'yes':
            raise RuntimeError('Correctness check failed: ' + line)
        times[fields[0]] = float(fields[2])
    if set(times) != set(STAGES) or any(t <= 0 or not np.isfinite(t) for t in times.values()):
        raise RuntimeError('Cannot read valid times for all six stages. Check the saved log.')
    return times


def chart(rows, sweep, ticks, xlabel, title, path):
    fig, ax = plt.subplots(figsize=(12, 6))
    positions = np.arange(len(ticks))
    width = .82 / len(STAGES)
    for i, stage in enumerate(STAGES):
        values = []
        for tick in ticks:
            match = [r for r in rows if r['sweep'] == sweep and r['stage'] == stage
                     and r['x'] == tick]
            if len(match) != 1:
                raise RuntimeError('Incomplete plot data')
            values.append(match[0]['speedup'])
        ax.bar(positions + (i - (len(STAGES)-1)/2)*width, values,
               width=width, label=LABELS[i], color=COLORS[i])
    ax.axhline(1, color='black', linestyle='--', linewidth=1, label='Naive reference (1×)')
    ax.set_xticks(positions, labels=list(map(str,ticks)))
    ax.set(xlabel=xlabel, ylabel='Speedup over naive', title=title, ylim=(0,None))
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.3)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01,1), frameon=False)
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=220)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    SCRIPT_DIR = Path(__file__).resolve().parent
    TASK1_ROOT = SCRIPT_DIR.parent
    parser.add_argument(
        '--root',
        type=Path,
        default=TASK1_ROOT,
        help='task1 directory'
    )
    parser.add_argument('--cpu', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=3, help='Whole harness runs per workload')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--csv', type=Path, help='Replot this script\'s CSV without running benchmarks')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    root = args.root.resolve()
    output = (args.output or (args.csv.parent / 'bar_plots' if args.csv else
              root / 'results' / ('all_speedups_' + datetime.now().strftime('%Y%m%d_%H%M%S')))).resolve()
    output.mkdir(parents=True, exist_ok=bool(args.csv))
    rows = []
    if args.csv:
        with args.csv.open(newline='') as stream:
            for raw in csv.DictReader(stream):
                rows.append(dict(raw, x=int(raw['x']), speedup=float(raw['speedup'])))
    else:
        logs = output / 'logs'
        logs.mkdir()
        env = dict(os.environ, LC_ALL='C')
        print('Building current sources with make -B...', flush=True)
        build = subprocess.run(['make', '-B'], cwd=root, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (logs / 'build.log').write_text(build.stdout)
        if build.returncode:
            raise RuntimeError(f'Build failed: {build.stdout}')
        workloads = [('image', n, n, 5) for n in SIZES]
        workloads += [('kernel', k, 2048, k) for k in KERNELS]
        settings = dict(cpu=args.cpu, repeats=args.repeats, seed=args.seed,
                        image_sizes=SIZES, fixed_kernel=5, kernel_sizes=KERNELS,
                        fixed_image=2048, speedup='Median of paired naive/stage time ratios',
                        timing_source='Rounded milliseconds printed by the assignment harness')
        (output / 'settings.json').write_text(json.dumps(settings, indent=2))
        fields = ['sweep','x','H','W','K','stage','time_ms','speedup','cpu','repeats','seed']
        cache = {}
        with (output / 'speedup_results.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for sweep, x, n, k in workloads:
                print(f'\n{sweep}: H=W={n}, K={k}', flush=True)
                if (n,k) not in cache:
                    samples = []
                    for repeat in range(args.repeats):
                        print(f'  run {repeat+1}/{args.repeats}', flush=True)
                        command = ['taskset','-c',str(args.cpu),str(root/'bin/conv'),
                                   'all',str(n),str(n),str(k),str(args.seed)]
                        result = subprocess.run(command,cwd=root,env=env,text=True,
                                                stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                        log = logs / f'n{n}_k{k}_run{repeat+1}.log'
                        log.write_text(result.stdout)
                        print(result.stdout, flush=True)
                        if result.returncode:
                            raise RuntimeError(f'Harness failed; see {log}')
                        samples.append(parse_table(result.stdout))
                    cache[n,k] = samples
                samples = cache[n,k]
                for stage in STAGES:
                    row = dict(sweep=sweep,x=x,H=n,W=n,K=k,stage=stage,
                               time_ms=statistics.median(s[stage] for s in samples),
                               speedup=statistics.median(s['naive']/s[stage] for s in samples),
                               cpu=args.cpu,repeats=args.repeats,seed=args.seed)
                    writer.writerow(row)
                    rows.append(row)
                stream.flush()
    chart(rows,'image',SIZES,'Image size (H = W)',
          'Convolution speedup vs image size (K = 5)',output/'speedup_vs_image_size')
    chart(rows,'kernel',KERNELS,'Kernel size (K)',
          'Convolution speedup vs kernel size (H = W = 2048)',output/'speedup_vs_kernel_size')
    print(f'\nSaved two charts in PNG and PDF format to {output}')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error))
