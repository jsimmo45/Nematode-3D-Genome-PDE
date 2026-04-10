#!/usr/bin/env python3
"""
plot_insulation_genome.py
=========================
Plot insulation scores across all Ascaris chromosomes for multiple
developmental stages in a stacked grid layout.  Each chromosome is a column,
each stage is a row, with post-PDE coordinates mapped back to germline space
and eliminated regions shaded red.

Autosomal traces are blue, sex chromosome (chrX) traces are magenta.
Post-PDE stages have gaps at eliminated regions where data is absent.
Optional moving-average smoothing for cleaner visualization.

Outputs:
  - Stacked genome-wide insulation plot (PNG + SVG)

Dependencies:
  numpy, matplotlib, scipy

Example usage:
  python plot_insulation_genome.py \\
      --elimination-bed data/AG_v50_eliminated_strict.bed \\
      --mapping-bed data/as_v50_prepde_to_postpde_coordinates.bed \\
      --chrom-sizes data/AG_v50_chrom_sizes.txt \\
      --pre-dir insulation_scores/prepde \\
      --post-dir insulation_scores/postpde \\
      --samples 0hr 48hr 60hr 5day 10day \\
      --display-names '0hr:1cell;48hr:2-4cell;60hr:4-8cell;5day:32-64cell;10day:L1' \\
      --smoothing 20 \\
      --output-dir results/

Input files:
  --elimination-bed: BED file of eliminated DNA regions.
  --mapping-bed: Pre-PDE to post-PDE coordinate mapping (4 columns).
  --chrom-sizes: Chromosome sizes (chr  length).
  --pre-dir / --post-dir: Directories with FAN-C insulation bedGraph files.
"""

import os
import sys
import glob
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d

# Publication-quality settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['svg.fonttype'] = 'none'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot insulation scores across Ascaris development')
    parser.add_argument('--elimination-bed', required=True,
                        help='BED file with eliminated DNA regions')
    parser.add_argument('--mapping-bed', required=True,
                        help='Pre-PDE to post-PDE coordinate mapping')
    parser.add_argument('--chrom-sizes', required=True,
                        help='Chromosome sizes file (chr  length)')
    parser.add_argument('--pre-dir', required=True,
                        help='Directory with pre-PDE insulation bedGraph files')
    parser.add_argument('--post-dir', required=True,
                        help='Directory with post-PDE insulation bedGraph files')
    parser.add_argument('--output-dir', default='insulation_plots',
                        help='Output directory')
    parser.add_argument('--smoothing', type=int, default=0,
                        help='Smoothing window in bins (0 = no smoothing)')
    parser.add_argument('--display-names', default='',
                        help='Display name mapping (name:label;name:label;...)')
    parser.add_argument('--samples', nargs='+', required=True,
                        help='Sample names to plot (e.g., 0hr 48hr 60hr 5day 10day)')
    return parser.parse_args()


args = parse_args()
ELIM_BED = args.elimination_bed
MAPPING_BED = args.mapping_bed
CHROM_SIZES = args.chrom_sizes
PRE_DIR = args.pre_dir
POST_DIR = args.post_dir
OUTPUT_DIR = args.output_dir
SMOOTHING_WINDOW = args.smoothing
DISPLAY_MAP_STR = args.display_names
SELECTED_SAMPLES = args.samples

print(f"Configuration:")
print(f"  Eliminated regions: {ELIM_BED}")
print(f"  Coordinate mapping: {MAPPING_BED}")
print(f"  Chromosome sizes: {CHROM_SIZES}")
print(f"  PRE-PDE directory: {PRE_DIR}")
print(f"  POST-PDE directory: {POST_DIR}")
print(f"  Output directory: {OUTPUT_DIR}")
print(f"  Smoothing window: {SMOOTHING_WINDOW} bins {'(disabled)' if SMOOTHING_WINDOW == 0 else ''}")
print(f"  Selected samples: {SELECTED_SAMPLES}")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parse display name mapping
display_map = {}
if DISPLAY_MAP_STR:
    for pair in DISPLAY_MAP_STR.split(';'):
        if ':' in pair:
            filename, display_name = pair.split(':', 1)
            display_map[filename] = display_name

# Load eliminated regions
elim = []
with open(ELIM_BED) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3 and re.match(r"^chr(\d+|X\d+)$", parts[0]):
            elim.append((parts[0], int(parts[1]), int(parts[2])))

# Load coordinate mapping
mapping = {}
pre_to_segs = {}
with open(MAPPING_BED) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 4:
            pre, s, e, post = parts[0], int(parts[1]), int(parts[2]), parts[3]
            mapping[post] = (pre, s, e)
            pre_to_segs.setdefault(pre, []).append((post, s, e))

# Load chromosome sizes
chr_lengths = {}
pre_chrs = []
with open(CHROM_SIZES) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2 and re.match(r"^chr(\d+|X\d+)$", parts[0]):
            chr_lengths[parts[0]] = int(parts[1])
            pre_chrs.append(parts[0])

pre_chrs.sort(key=lambda c: (100 if c.startswith('chrX') else 0, int(re.search(r"\d+", c).group())))

# Find files for selected samples
files = []
for samp in SELECTED_SAMPLES:
    pre_pattern = os.path.join(PRE_DIR, f"AG.v50.{samp}.*.insulation.bedGraph")
    pre_files = glob.glob(pre_pattern)
    post_pattern = os.path.join(POST_DIR, f"AG.v50_post.{samp}.*.insulation.bedGraph")
    post_files = glob.glob(post_pattern)
    
    print(f"\nLooking for {samp}:")
    print(f"  PRE pattern: {pre_pattern}")
    print(f"  PRE files found: {pre_files}")
    print(f"  POST pattern: {post_pattern}")
    print(f"  POST files found: {post_files}")

    if pre_files:
        files.append((samp, pre_files[0], True))
        print(f"  -> Added PRE-PDE: {pre_files[0]}")
    elif post_files:
        files.append((samp, post_files[0], False))
        print(f"  -> Added POST-PDE: {post_files[0]}")
    else:
        print(f"  -> WARNING: No files found for {samp}")

if not files:
    print("ERROR: No files found! Exiting.")
    sys.exit(1)

# Helper: check if position is in eliminated region
def is_eliminated(pos, chrom):
    for e_chr, e_start, e_end in elim:
        if e_chr == chrom and e_start <= pos <= e_end:
            return True
    return False

# Updated loader
def load_bedgraph_data(filepath, target_chr, is_pre):
    data = []
    with open(filepath) as f:
        for line in f:
            if line.startswith('track') or line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            chrom, start, end, value = parts[0], int(parts[1]), int(parts[2]), parts[3]
            if value in ('NA', 'nan'):
                continue
            try:
                value = float(value)
            except ValueError:
                continue
            midpoint = (start + end) // 2
            mapped_pos = None

            if is_pre:
                if chrom == target_chr:
                    mapped_pos = midpoint
            else:
                if chrom in mapping:
                    pre_chr, seg_s, seg_e = mapping[chrom]
                    if pre_chr == target_chr:
                        mapped_pos = seg_s + midpoint
                elif chrom == target_chr:
                    segs = pre_to_segs.get(chrom, [])
                    if len(segs) == 1:
                        _, seg_s, _ = segs[0]
                        mapped_pos = seg_s + midpoint
                    else:
                        mapped_pos = midpoint

            if mapped_pos is None:
                continue
            if not is_pre and is_eliminated(mapped_pos, target_chr):
                continue
            data.append((mapped_pos, value))
    return np.array(data) if data else np.array([]).reshape(0, 2)

# Smoothing
def smooth_data(data, window):
    if window <= 1 or len(data) < window:
        return data
    sorted_idx = np.argsort(data[:, 0])
    sorted_data = data[sorted_idx]
    smoothed_values = uniform_filter1d(sorted_data[:, 1], size=window, mode='nearest')
    result = sorted_data.copy()
    result[:, 1] = smoothed_values
    return result

# Group chromosomes into rows of 5
n_cols = 5
n_chr_rows = (len(pre_chrs) + n_cols - 1) // n_cols
chr_rows = [pre_chrs[i * n_cols:(i + 1) * n_cols] for i in range(n_chr_rows)]

# Calculate total rows
total_rows = n_chr_rows * len(files)
row_height = 0.85 / total_rows
row_gap_within = 0.005
row_gap_between = 0.025

# Create large figure
fig_height = 2.0 * total_rows + 1.5
fig = plt.figure(figsize=(20, fig_height))
print(f"\nCreating plot with {total_rows} total rows")

plot_row_idx = 0

for chr_row_idx, row_chrs in enumerate(chr_rows):
    if not row_chrs:
        continue
    print(f"\nChromosome group {chr_row_idx + 1}: {row_chrs}")
    for sample_idx, (sample_name, filepath, is_pre) in enumerate(files):
        print(f"  Row {plot_row_idx + 1}/{total_rows}: {sample_name}")
        row_widths = [chr_lengths[c] for c in row_chrs]
        cumulative_gap = sum(row_gap_between if (i % len(files) == len(files) - 1) else row_gap_within for i in range(plot_row_idx))
        gs = gridspec.GridSpec(
            1, len(row_chrs),
            left=0.08, right=0.95,
            top=0.94 - plot_row_idx * row_height - cumulative_gap,
            bottom=0.94 - (plot_row_idx + 1) * row_height - cumulative_gap,
            width_ratios=row_widths, wspace=0.25
        )

        for col_idx, chrom in enumerate(row_chrs):
            ax = fig.add_subplot(gs[0, col_idx])
            chr_len = chr_lengths[chrom]
            line_color = 'magenta' if chrom.startswith('chrX') else 'steelblue'
            data = load_bedgraph_data(filepath, chrom, is_pre)

            if len(data) > 0:
                order = np.argsort(data[:, 0])
                data = data[order]

                if not is_pre:
                    mask = np.array([not is_eliminated(p, chrom) for p in data[:, 0]])
                    data = data[mask]
                if data.size == 0:
                    continue

                if is_pre:
                    if SMOOTHING_WINDOW > 1:
                        data = smooth_data(data, SMOOTHING_WINDOW)
                    ax.plot(data[:, 0], data[:, 1], color=line_color, lw=2.5, alpha=0.9)
                else:
                    diffs = np.diff(data[:, 0]) if len(data) > 1 else np.array([1])
                    pos_diffs = diffs[diffs > 0]
                    median_diff = float(np.median(pos_diffs)) if pos_diffs.size > 0 else 1.0
                    gap_threshold = median_diff * 3.0
                    chrom_elim_ranges = [(s, e) for e_chr, s, e in elim if e_chr == chrom]
                    segments, cur_x, cur_y = [], [], []
                    prev_pos = None
                    for pos, val in data:
                        should_break = False
                        if prev_pos is not None:
                            if (pos - prev_pos) > gap_threshold:
                                should_break = True
                            else:
                                for e_s, e_e in chrom_elim_ranges:
                                    if prev_pos < e_s < pos or prev_pos < e_e < pos:
                                        should_break = True
                                        break
                        if should_break and cur_x:
                            segments.append((np.array(cur_x), np.array(cur_y)))
                            cur_x, cur_y = [], []
                        cur_x.append(pos)
                        cur_y.append(val)
                        prev_pos = pos
                    if cur_x:
                        segments.append((np.array(cur_x), np.array(cur_y)))

                    for seg_x, seg_y in segments:
                        if SMOOTHING_WINDOW > 1 and len(seg_y) >= SMOOTHING_WINDOW:
                            seg_y = uniform_filter1d(seg_y, size=SMOOTHING_WINDOW, mode='nearest')
                        ax.plot(seg_x, seg_y, color=line_color, lw=2.5, alpha=0.9)

            for e_chr, e_start, e_end in elim:
                if e_chr == chrom:
                    ax.axvspan(e_start, e_end, color='red', alpha=0.3)
            ax.axhline(0, linestyle=':', color='gray', alpha=0.5, lw=1.5)
            ax.set_ylim(-2, 1)
            ax.set_xlim(0, chr_len)

            if sample_idx == len(files) - 1:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.1f}"))
            else:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: ""))
                ax.tick_params(axis='x', length=0)

            if sample_idx == len(files) - 1:
                fontsize = 18 if chrom in pre_to_segs and len(pre_to_segs[chrom]) > 1 else 16
                ax.set_xlabel(chrom, fontweight='bold', fontsize=fontsize)
            else:
                ax.set_xlabel('')

            if sample_idx == 0 and chrom in pre_to_segs and len(pre_to_segs[chrom]) > 1:
                y_min, y_max = ax.get_ylim()
                for post_chr, offset, end_pos in pre_to_segs[chrom]:
                    mid_pos = offset + (end_pos - offset) / 2
                    ax.text(mid_pos, y_max + (y_max - y_min) * 0.05, post_chr,
                            ha='center', va='bottom', fontsize=14)

            if col_idx == 0:
                sample_label = display_map.get(sample_name, sample_name)
                ax.set_ylabel(f"{sample_label}\nInsulation Score", fontweight='bold', fontsize=14)
            else:
                ax.set_ylabel('')
        plot_row_idx += 1

legend_handles = [
    Line2D([0], [0], color='steelblue', lw=3, label='Autosomes'),
    Line2D([0], [0], color='magenta', lw=3, label='Sex chromosomes')
]
fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.995),
           ncol=2, frameon=True, fontsize=14, columnspacing=2.0, handlelength=2.5)

title_text = r"Insulation score across $\mathbf{\mathit{Ascaris}}$ development"
if SMOOTHING_WINDOW > 1:
    title_text += f" (smoothed: {SMOOTHING_WINDOW}-bin window)"
fig.suptitle(title_text, fontsize=20, y=0.998, fontweight='bold')

output_names = [display_map.get(s, s).replace('-', '').replace(' ', '') for s in SELECTED_SAMPLES]
output_prefix = '_'.join(output_names)
smooth_suffix = f"_smooth{SMOOTHING_WINDOW}" if SMOOTHING_WINDOW > 1 else ""
output_file = os.path.join(OUTPUT_DIR, f'insulation_stacked_{output_prefix}{smooth_suffix}')

fig.savefig(f'{output_file}.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{output_file}.svg', bbox_inches='tight')

print(f'\n✓ Plot complete: {output_file}.png')
print(f'✓ Plot complete: {output_file}.svg')
print(f'Dimensions: {fig.get_size_inches()[0]:.1f}\" × {fig.get_size_inches()[1]:.1f}\" ({total_rows} rows)')
print(f'Y-axis range: -2 to +1 (fixed for all plots)')

