#!/usr/bin/env python3
"""Heatmaps of measured image-size/tile-size pairs. Keep plot_tiles_1d.py alongside."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from plot_tiles_1d import read_data, save, LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path, help='tile_results.csv from sweep_tiles.py')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    rows = read_data(args.csv)
    output = args.output_dir or args.csv.parent / 'plots_2d'
    output.mkdir(parents=True, exist_ok=True)
    for kernel in sorted({r['K'] for r in rows}):
        selected = [r for r in rows if r['K'] == kernel]
        sizes = sorted({r['H'] for r in selected})
        tiles = sorted({r['tile_size'] for r in selected})
        for metric, label in LABELS.items():
            matrix = np.full((len(sizes), len(tiles)), np.nan)
            for row in selected:
                matrix[sizes.index(row['H']), tiles.index(row['tile_size'])] = row[metric]
            cmap = plt.get_cmap('RdBu' if metric == 'speedup' else 'viridis_r').copy()
            cmap.set_bad('#e5e5e5')
            norm = None
            if metric == 'speedup':
                # White means equal to naive, red slower, blue faster.
                norm = TwoSlopeNorm(vmin=min(float(np.nanmin(matrix)), .99),
                                    vcenter=1, vmax=max(float(np.nanmax(matrix)), 1.01))
            fig, ax = plt.subplots(figsize=(max(9,len(tiles)*.95), 5.5))
            im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, aspect='auto')
            ax.set_xticks(range(len(tiles)), labels=list(map(str,tiles)))
            ax.set_yticks(range(len(sizes)), labels=list(map(str,sizes)))
            ax.set(xlabel='Tile side length', ylabel='Image size (H = W)',
                   title=f'{label} (K={kernel}) — grey cells were not measured')
            for i in range(len(sizes)):
                for j in range(len(tiles)):
                    value = matrix[i,j]
                    if np.isfinite(value):
                        rgba = im.cmap(im.norm(value))
                        color = 'black' if sum(rgba[:3])/3 > .55 else 'white'
                        text = f'{value:.3f}' + ('×' if metric == 'speedup' else '')
                        ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)
            fig.colorbar(im, ax=ax, label=label)
            save(fig, output / f'{metric}_heatmap_k{kernel}')
    print(f'Saved PNG and PDF heatmaps to {output}')
    print('Speedup is relative to naive; MPKI is the measured tile MPKI.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(str(error))
