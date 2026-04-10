#!/usr/bin/env python3
"""
plot_eigenvector_genome.py
==========================
Plot eigenvector 1 (EV1 / PC1 compartment signal) across all Ascaris
chromosomes for multiple developmental stages in a stacked grid layout.

Each chromosome is a column, each stage is a row.  Post-PDE stages have
coordinates mapped back to germline space with gaps at eliminated regions
(shaded red).  Autosomal traces are blue, sex chromosome (chrX) traces are
magenta.

EV1 sign correction is handled via per-sample-chromosome flip specifications
(--flip-combos), since eigenvector sign is arbitrary and must be manually
corrected for biological consistency across stages.

Outputs:
  - Stacked genome-wide eigenvector plot (PNG + SVG)

Dependencies:
  numpy, matplotlib, scipy

Example usage:
  python plot_eigenvector_genome.py \\
      --elimination-bed data/AG_v50_eliminated_strict.bed \\
      --mapping-bed data/as_v50_prepde_to_postpde_coordinates.bed \\
      --chrom-sizes data/AG_v50_chrom_sizes.txt \\
      --pre-dir data/eigenvectors/ascaris/prepde \\
      --post-dir data/eigenvectors/ascaris/postpde \\
      --samples as_0hr_iced_100kb as_48hr_iced_100kb as_60hr_iced_100kb \\
               as_5day_iced_100kb as_10day_iced_100kb \\
      --display-names 'as_0hr_iced_100kb:1cell;as_48hr_iced_100kb:2-4cell;as_60hr_iced_100kb:4-8cell;as_5day_iced_100kb:32-64cell;as_10day_iced_100kb:L1' \\
      --flip-combos '48hr,chr01;60hr,chr01;0hr,chr03' \\
      --smoothing 3 \\
      --output-dir results/

Input files:
  --elimination-bed: BED of eliminated DNA regions.
  --mapping-bed: Pre-PDE to post-PDE coordinate mapping (4 cols).
  --chrom-sizes: Chromosome sizes (chr  length).
  --pre-dir / --post-dir: Directories with FAN-C eigenvector files
      (FANC format: chr  start  end  value).
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
        description='Plot eigenvector 1 across Ascaris development')
    parser.add_argument('--elimination-bed', required=True,
                        help='BED file with eliminated DNA regions')
    parser.add_argument('--mapping-bed', required=True,
                        help='Pre-PDE to post-PDE coordinate mapping')
    parser.add_argument('--chrom-sizes', required=True,
                        help='Chromosome sizes file (chr  length)')
    parser.add_argument('--pre-dir', required=True,
                        help='Directory with pre-PDE eigenvector files')
    parser.add_argument('--post-dir', required=True,
                        help='Directory with post-PDE eigenvector files')
    parser.add_argument('--output-dir', default='ev1_stacked_output',
                        help='Output directory')
    parser.add_argument('--smoothing', type=int, default=0,
                        help='Smoothing window in bins (0 = no smoothing)')
    parser.add_argument('--flip-combos', default='',
                        help='EV1 sign corrections (sample,chr;sample,chr;...)')
    parser.add_argument('--display-names', default='',
                        help='Display name mapping (name:label;name:label;...)')
    parser.add_argument('--samples', nargs='+', required=True,
                        help='Sample filenames to plot')
    return parser.parse_args()


args = parse_args()
ELIM_BED = args.elimination_bed
MAPPING_BED = args.mapping_bed
CHROM_SIZES = args.chrom_sizes
PRE_DIR = args.pre_dir
POST_DIR = args.post_dir
OUTPUT_DIR = args.output_dir
SMOOTHING_WINDOW = args.smoothing
FLIP_COMBOS_STR = args.flip_combos
DISPLAY_MAP_STR = args.display_names
SELECTED_SAMPLES = args.samples

# Parse flip combinations
flip_set = set()
if FLIP_COMBOS_STR and FLIP_COMBOS_STR != "NONE":
    for combo in FLIP_COMBOS_STR.split(';'):
        if combo.strip():
            flip_set.add(combo.strip())

# Parse display name mapping
display_map = {}
if DISPLAY_MAP_STR:
    for pair in DISPLAY_MAP_STR.split(';'):
        if ':' in pair:
            filename, display_name = pair.split(':', 1)
            display_map[filename] = display_name

print(f"Configuration:")
print(f"  Eliminated regions: {ELIM_BED}")
print(f"  Coordinate mapping: {MAPPING_BED}")
print(f"  Chromosome sizes: {CHROM_SIZES}")
print(f"  PRE-PDE directory: {PRE_DIR}")
print(f"  POST-PDE directory: {POST_DIR}")
print(f"  Output directory: {OUTPUT_DIR}")
print(f"  Smoothing window: {SMOOTHING_WINDOW} bins {'(disabled)' if SMOOTHING_WINDOW == 0 else ''}")
print(f"  Flip combinations: {flip_set if flip_set else 'None'}")
print(f"  Selected samples: {SELECTED_SAMPLES}")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    pre_files = glob.glob(os.path.join(PRE_DIR, f"{samp}*.matrix.eigenvector"))
    post_files = glob.glob(os.path.join(POST_DIR, f"{samp}*.matrix.eigenvector"))
    
    print(f"\nLooking for {samp}:")
    print(f"  PRE files found: {pre_files}")
    print(f"  POST files found: {post_files}")

    if pre_files:
        files.append((samp, pre_files[0], True))
        print(f"  -> Added PRE-PDE: {pre_files[0]}")
    elif post_files:
        files.append((samp, post_files[0], False))
        print(f"  -> Added POST-PDE: {post_files[0]}")
    else:
        print(f"  -> WARNING: No files found for {samp}")

print(f"\nTotal files to process: {len(files)}")
for i, (samp, filepath, is_pre) in enumerate(files):
    print(f"  {i+1}. {samp} ({'PRE' if is_pre else 'POST'}-PDE): {filepath}")

if not files:
    print("ERROR: No files found! Exiting.")
    sys.exit(1)

# Data loading function
def load_simple_data(filepath, target_chr):
    data = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                chrom, start, end, value = parts[0], int(parts[1]), int(parts[2]), float(parts[3])

                if chrom == target_chr:
                    data.append((start, value))
                elif chrom in mapping and mapping[chrom][0] == target_chr:
                    offset = mapping[chrom][1]
                    data.append((offset + start, value))

    return np.array(data) if data else np.array([]).reshape(0, 2)

# Smoothing function
def smooth_data(data, window):
    """Apply moving average smoothing to data"""
    if window <= 1 or len(data) < window:
        return data
    
    sorted_idx = np.argsort(data[:, 0])
    sorted_data = data[sorted_idx]
    smoothed_values = uniform_filter1d(sorted_data[:, 1], size=window, mode='nearest')
    result = sorted_data.copy()
    result[:, 1] = smoothed_values
    return result

# Check if position is in eliminated region
def is_eliminated(pos, chrom):
    for e_chr, e_start, e_end in elim:
        if e_chr == chrom and e_start <= pos <= e_end:
            return True
    return False

# Check if this sample-chromosome combo needs flipping
def should_flip(sample_name, chrom):
    # Get display name if available
    display_name = display_map.get(sample_name, sample_name.replace('as_', '').replace('_iced_100kb', ''))
    
    # Try with display name
    combo = f"{display_name},{chrom}"
    if combo in flip_set:
        return True
    
    # Try with shortened sample name
    combo = f"{sample_name.replace('as_', '').replace('_iced_100kb', '')},{chrom}"
    if combo in flip_set:
        return True
    
    # Try with full sample name
    combo_full = f"{sample_name},{chrom}"
    if combo_full in flip_set:
        return True
    
    return False

# Group chromosomes into rows of 5
n_cols = 5
n_chr_rows = (len(pre_chrs) + n_cols - 1) // n_cols

chr_rows = []
for i in range(n_chr_rows):
    start_idx = i * n_cols
    end_idx = min(start_idx + n_cols, len(pre_chrs))
    chr_rows.append(pre_chrs[start_idx:end_idx])

# Calculate total number of rows
total_rows = n_chr_rows * len(files)
row_height = 0.85 / total_rows
row_gap_within = 0.005  # Very small gap within chromosome group
row_gap_between = 0.025  # Larger gap between chromosome groups

# Create large figure
fig_height = 2.0 * total_rows + 1.5
fig = plt.figure(figsize=(20, fig_height))

print(f"\nCreating plot with {total_rows} total rows ({n_chr_rows} chromosome rows × {len(files)} samples)")

# Plot: for each chromosome row, repeat for each sample
plot_row_idx = 0

for chr_row_idx, row_chrs in enumerate(chr_rows):
    if not row_chrs:
        continue
    
    print(f"\nChromosome group {chr_row_idx + 1}: {row_chrs}")
    
    for sample_idx, (sample_name, filepath, is_pre) in enumerate(files):
        print(f"  Row {plot_row_idx + 1}/{total_rows}: {sample_name}")
        
        row_widths = [chr_lengths[chrom] for chrom in row_chrs]

        # Determine gap: larger between chromosome groups, smaller within
        is_last_in_group = (sample_idx == len(files) - 1)
        current_gap = row_gap_between if is_last_in_group else row_gap_within
        
        # Calculate cumulative gap up to this row
        cumulative_gap = 0
        for i in range(plot_row_idx):
            row_chr_idx = i // len(files)
            row_sample_idx = i % len(files)
            if row_sample_idx == len(files) - 1:
                cumulative_gap += row_gap_between
            else:
                cumulative_gap += row_gap_within

        gs = gridspec.GridSpec(
            1, len(row_chrs),
            left=0.08, 
            right=0.95,
            top=0.94 - plot_row_idx * row_height - cumulative_gap,
            bottom=0.94 - (plot_row_idx + 1) * row_height - cumulative_gap,
            width_ratios=row_widths,
            wspace=0.25
        )

        for col_idx, chrom in enumerate(row_chrs):
            ax = fig.add_subplot(gs[0, col_idx])
            chr_len = chr_lengths[chrom]

            # Determine color based on chromosome type
            line_color = 'magenta' if chrom.startswith('chrX') else 'steelblue'

            # Load data for this sample and chromosome
            data = load_simple_data(filepath, chrom)

            # Check if we need to flip this combination
            needs_flip = should_flip(sample_name, chrom)
            if needs_flip:
                print(f"    -> Flipping y-axis for {sample_name},{chrom}")

            if len(data) > 0:
                # Apply smoothing if requested
                if SMOOTHING_WINDOW > 1:
                    data = smooth_data(data, SMOOTHING_WINDOW)

                # Apply flip if needed
                if needs_flip:
                    data[:, 1] = -data[:, 1]

                if is_pre:
                    ax.plot(data[:,0], data[:,1], color=line_color, lw=2.5, alpha=0.9)
                else:
                    segments = []
                    current_segment_x = []
                    current_segment_y = []

                    for pos, val in data:
                        if is_eliminated(pos, chrom):
                            if current_segment_x:
                                segments.append((np.array(current_segment_x), np.array(current_segment_y)))
                                current_segment_x = []
                                current_segment_y = []
                        else:
                            current_segment_x.append(pos)
                            current_segment_y.append(val)

                    if current_segment_x:
                        segments.append((np.array(current_segment_x), np.array(current_segment_y)))

                    for seg_x, seg_y in segments:
                        ax.plot(seg_x, seg_y, color=line_color, lw=2.5, alpha=0.9)

            # Add eliminated regions as red shading
            for e_chr, e_start, e_end in elim:
                if e_chr == chrom:
                    ax.axvspan(e_start, e_end, color='red', alpha=0.3)

            # Add reference line
            ax.axhline(0, linestyle=':', color='gray', alpha=0.5, lw=1.5)

            # Set FIXED y-axis limits
            ax.set_ylim(-0.3, 0.3)

            # Set x-axis limits
            ax.set_xlim(0, chr_len)
            
            # X-axis ticks and labels - only for LAST sample in group
            is_last_sample = (sample_idx == len(files) - 1)
            if is_last_sample:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.1f}"))
            else:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: ""))
                ax.tick_params(axis='x', length=0)  # Hide tick marks too

            # Chromosome label on BOTTOM - only for LAST sample in group
            if is_last_sample:
                fontsize = 18 if chrom in pre_to_segs and len(pre_to_segs[chrom]) > 1 else 16  # Line for bottom chromosome label size
                ax.set_xlabel(chrom, fontweight='bold', fontsize=fontsize)
            else:
                ax.set_xlabel('')
            
            # Add post-PDE labels at TOP - only for FIRST sample in group
            is_first_sample = (sample_idx == 0)
            if is_first_sample and chrom in pre_to_segs and len(pre_to_segs[chrom]) > 1:
                y_min, y_max = ax.get_ylim()
                for post_chr, offset, end_pos in pre_to_segs[chrom]:
                    mid_pos = offset + (end_pos - offset) / 2
                    ax.text(mid_pos, y_max + (y_max - y_min) * 0.05, post_chr,
                           ha='center', va='bottom', fontsize=14)  # Line for top post-PDE label size

            # Y-label: sample name on leftmost plot
            if col_idx == 0:
                # Use display name if available, otherwise clean up filename
                if sample_name in display_map:
                    sample_label = display_map[sample_name]
                else:
                    sample_label = sample_name.replace('as_', '').replace('_iced_100kb', '')
                ax.set_ylabel(f"{sample_label}\nEV1", fontweight='bold', fontsize=14)  # Line for sample label size
            else:
                ax.set_ylabel('')

        plot_row_idx += 1

# Create legend
legend_handles = [
    Line2D([0], [0], color='steelblue', lw=3, label='Autosomes'),
    Line2D([0], [0], color='magenta', lw=3, label='Sex chromosomes')
]
fig.legend(handles=legend_handles, 
          loc='upper center',
          bbox_to_anchor=(0.5, 0.995),
          ncol=2,
          frameon=True, 
          fontsize=14,
          columnspacing=2.0,
          handlelength=2.5)

# Title
title_text = r"Eigenvector 1 across $\mathbf{\mathit{Ascaris}}$ development"
if SMOOTHING_WINDOW > 1:
    title_text += f" (smoothed: {SMOOTHING_WINDOW}-bin window)"
fig.suptitle(title_text, fontsize=20, y=0.998, fontweight='bold')

# Save figure
output_names = []
for samp in SELECTED_SAMPLES:
    if samp in display_map:
        output_names.append(display_map[samp].replace('-', '').replace(' ', ''))
    else:
        output_names.append(samp.replace('as_', '').replace('_iced_100kb', ''))
output_prefix = '_'.join(output_names)
smooth_suffix = f"_smooth{SMOOTHING_WINDOW}" if SMOOTHING_WINDOW > 1 else ""
output_file = os.path.join(OUTPUT_DIR, f'eigen_stacked_{output_prefix}{smooth_suffix}')

fig.savefig(f'{output_file}.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{output_file}.svg', bbox_inches='tight')
print(f'\n✓ Plot complete: {output_file}.png')
print(f'✓ Plot complete: {output_file}.svg')
print(f'\nDimensions: {fig.get_size_inches()[0]:.1f}" × {fig.get_size_inches()[1]:.1f}" ({total_rows} rows)')
print(f'Y-axis range: -0.3 to +0.3 (fixed for all plots)')
if flip_set:
    print(f'Flipped combinations: {flip_set}')
