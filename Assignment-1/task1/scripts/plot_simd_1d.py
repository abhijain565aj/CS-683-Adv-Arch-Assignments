#!/usr/bin/env python3
"""Plot the CSV produced by sweep_simd_widths.py; no measurements are run."""
import argparse
import csv
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_data(path):
    keys = ['H', 'W', 'K', 'simd_width']
    metrics = ['naive_instructions', 'simd_instructions', 'speedup']
    rows, seen = [], set()
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not set(keys + metrics) <= set(reader.fieldnames or []):
            raise ValueError('CSV must contain: ' + ', '.join(keys + metrics))
        for line, raw in enumerate(reader, 2):
            row = {k: int(raw[k]) for k in keys}
            if row['H'] != row['W'] or any(row[k] <= 0 for k in keys):
                raise ValueError(f'Invalid image or width at row {line}')
            for metric in metrics:
                row[metric] = float(raw[metric])
                if not math.isfinite(row[metric]) or row[metric] <= 0:
                    raise ValueError(f'Invalid {metric} at row {line}')
            row['instruction_reduction'] = row['naive_instructions']/row['simd_instructions']
            identity = tuple(row[k] for k in keys)
            if identity in seen:
                raise ValueError(f'Duplicate measurement at row {line}; use one sweep CSV.')
            seen.add(identity)
            rows.append(row)
    if not rows:
        raise ValueError('CSV has no measurements')
    return rows


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=200)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def decorate(ax, sizes, ylabel, title):
    ax.set_xscale('log', base=2)
    ax.set_xticks(sizes, labels=list(map(str, sizes)))
    ax.set(xlabel='Image size (H = W)', ylabel=ylabel, title=title)
    ax.grid(alpha=.3)
    ax.legend()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    rows = read_data(args.csv)
    output = args.output_dir or args.csv.parent / 'plots_1d'
    output.mkdir(parents=True, exist_ok=True)
    colors = {128:'tab:blue', 256:'tab:orange', 512:'tab:green'}
    for kernel in sorted({r['K'] for r in rows}):
        subset = [r for r in rows if r['K'] == kernel]
        widths = sorted({r['simd_width'] for r in subset})
        sizes = sorted({r['H'] for r in subset})
        for metric, ylabel in [('speedup','Speedup over naive'),
                               ('simd_instructions','Instructions'),
                               ('instruction_reduction','Instruction reduction (naive / SIMD)')]:
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for width in widths:
                pts = sorted((r for r in subset if r['simd_width']==width), key=lambda r:r['H'])
                x = [r['H'] for r in pts]
                ax.plot(x, [r[metric] for r in pts], marker='o',
                        color=colors.get(width), label=f'{width}-bit SIMD')
                if metric == 'simd_instructions':
                    ax.plot(x, [r['naive_instructions'] for r in pts], linestyle='--',
                            color=colors.get(width), label=f'naive ({width}-bit run)')
            if metric != 'simd_instructions':
                ax.axhline(1, color='black', linestyle='--', label='naive (1×)')
            decorate(ax, sizes, ylabel, f'{ylabel} vs image size (K={kernel})')
            save(fig, output / f'all_widths_{metric}_k{kernel}')
            for width in widths:
                pts = sorted((r for r in subset if r['simd_width']==width), key=lambda r:r['H'])
                x = [r['H'] for r in pts]
                fig, ax = plt.subplots(figsize=(8,5))
                ax.plot(x, [r[metric] for r in pts], marker='o', label=f'{width}-bit SIMD')
                if metric == 'simd_instructions':
                    ax.plot(x, [r['naive_instructions'] for r in pts], color='black',
                            linestyle='--', marker='o', label='naive')
                else:
                    ax.axhline(1, color='black', linestyle='--', label='naive (1×)')
                decorate(ax, x, ylabel, f'{width}-bit SIMD vs naive (K={kernel})')
                save(fig, output / f'width{width}_{metric}_k{kernel}')
    print(f'Saved PNG and PDF plots to {output}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))
