#!/usr/bin/env python3
"""Line plots from sweep_tiles.py CSV. No benchmarks are run."""
import argparse
import csv
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {'l1_mpki': 'L1-D MPKI', 'speedup': 'Speedup over naive'}


def read_data(path):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'H', 'W', 'K', 'tile_size', 'l1_mpki', 'speedup'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('CSV must contain: ' + ', '.join(sorted(required)))
        rows = []
        seen = set()
        for line, raw in enumerate(reader, 2):
            row = {key: int(raw[key]) for key in ('H', 'W', 'K', 'tile_size')}
            for key in LABELS:
                row[key] = float(raw[key])
                if not math.isfinite(row[key]) or row[key] < 0:
                    raise ValueError(f'Invalid {key} at row {line}')
            if row['H'] != row['W'] or min(row[k] for k in ('H','K','tile_size')) <= 0:
                raise ValueError(f'Expected positive square image at row {line}')
            identity = (row['H'], row['K'], row['tile_size'])
            if identity in seen:
                raise ValueError(f'Duplicate pair at row {line}; select one sweep CSV.')
            seen.add(identity)
            rows.append(row)
    if not rows:
        raise ValueError('CSV is empty')
    return rows


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=200)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path, help='tile_results.csv from sweep_tiles.py')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    rows = read_data(args.csv)
    output = args.output_dir or args.csv.parent / 'plots_1d'
    output.mkdir(parents=True, exist_ok=True)
    for kernel in sorted({r['K'] for r in rows}):
        selected = [r for r in rows if r['K'] == kernel]
        sizes = sorted({r['H'] for r in selected})
        for metric, label in LABELS.items():
            # Combined plot: each curve has its own measured tile-size range.
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for size in sizes:
                points = sorted((r for r in selected if r['H'] == size), key=lambda r:r['tile_size'])
                ax.plot([r['tile_size'] for r in points], [r[metric] for r in points],
                        marker='o', label=f'{size} × {size}')
            if metric == 'speedup':
                ax.axhline(1, color='black', linestyle='--', label='naive (1×)')
            ticks = sorted({r['tile_size'] for r in selected})
            ax.set_xscale('log', base=2)
            ax.set_xticks(ticks, labels=list(map(str,ticks)))
            ax.set(xlabel='Tile side length', ylabel=label, title=f'{label} vs tile size (K={kernel})')
            ax.grid(alpha=.3)
            ax.legend(title='Image size')
            save(fig, output / f'all_sizes_{metric}_k{kernel}')
            for size in sizes:
                points = sorted((r for r in selected if r['H'] == size), key=lambda r:r['tile_size'])
                x, y = [r['tile_size'] for r in points], [r[metric] for r in points]
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(x, y, marker='o', label='tile')
                if metric == 'speedup':
                    ax.axhline(1, color='black', linestyle='--', label='naive (1×)')
                ax.set_xscale('log', base=2)
                ax.set_xticks(x, labels=list(map(str,x)))
                ax.set(xlabel='Tile side length', ylabel=label,
                       title=f'{label}: {size} × {size}, K={kernel}')
                ax.grid(alpha=.3)
                ax.legend()
                save(fig, output / f'n{size}_{metric}_k{kernel}')
    print(f'Saved PNG and PDF plots to {output}')
    print('MPKI is for tile only: the sweep CSV has no naive MPKI measurements.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(str(error))
