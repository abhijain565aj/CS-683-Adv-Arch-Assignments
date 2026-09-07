#!/usr/bin/env python3
"""SIMD heatmaps. Keep plot_simd_1d.py in the same directory."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from plot_simd_1d import read_data, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    rows = read_data(args.csv)
    output = args.output_dir or args.csv.parent / 'plots_2d'
    output.mkdir(parents=True, exist_ok=True)
    for kernel in sorted({r['K'] for r in rows}):
        subset = [r for r in rows if r['K']==kernel]
        sizes = sorted({r['H'] for r in subset})
        widths = sorted({r['simd_width'] for r in subset})
        for metric, label in [('speedup','Speedup over naive'),
                              ('simd_instructions','SIMD instructions'),
                              ('instruction_reduction','Instruction reduction (naive / SIMD)')]:
            matrix = np.full((len(sizes),len(widths)), np.nan)
            for row in subset:
                matrix[sizes.index(row['H']),widths.index(row['simd_width'])] = row[metric]
            ratio = metric != 'simd_instructions'
            cmap = plt.get_cmap('RdBu' if ratio else 'viridis_r').copy()
            cmap.set_bad('#e5e5e5')
            norm = None
            if ratio:
                norm = TwoSlopeNorm(vmin=min(float(np.nanmin(matrix)),.99), vcenter=1,
                                    vmax=max(float(np.nanmax(matrix)),1.01))
            fig, ax = plt.subplots(figsize=(7.5,5.5))
            im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, aspect='auto')
            ax.set_xticks(range(len(widths)), labels=list(map(str,widths)))
            ax.set_yticks(range(len(sizes)), labels=list(map(str,sizes)))
            ax.set(xlabel='SIMD register width (bits)', ylabel='Image size (H = W)',
                   title=f'{label} (K={kernel})')
            for i in range(len(sizes)):
                for j in range(len(widths)):
                    value = matrix[i,j]
                    if np.isfinite(value):
                        color = 'black' if sum(im.cmap(im.norm(value))[:3])/3 > .55 else 'white'
                        label_value = f'{value:.3f}×' if ratio else f'{value:,.0f}'
                        ax.text(j,i,label_value,ha='center',va='center',color=color,fontsize=9)
                    else:
                        ax.text(j,i,'not measured',ha='center',va='center',fontsize=9)
            fig.colorbar(im, ax=ax, label=label)
            save(fig, output / f'{metric}_heatmap_k{kernel}')
    print(f'Saved PNG and PDF heatmaps to {output}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))
