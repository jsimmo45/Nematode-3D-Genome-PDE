#!/usr/bin/env python3
"""
plot_ev1_single.py
==================
Plot eigenvector 1 (EV1) values across developmental stages for individual
Ascaris chromosomes.  Each stage is rendered as a separate panel stacked
vertically, sharing an x-axis (genomic position).  Eliminated DNA regions
are shaded red, and post-PDE stages are plotted with gaps at eliminated
loci.

EV1 sign is arbitrary in PCA, so per-chromosome, per-stage sign corrections
are specified via the FLIP_STAGES dictionary at the top of the script.
The y-axis is automatically centered at 0 with symmetric limits across all
stages.

Outputs (per chromosome):
  - PNG at 600 DPI
  - SVG with editable text for Adobe Illustrator

Dependencies:
  numpy, matplotlib

Example usage:
  python plot_ev1_single.py \
      --chromosomes chr01 chrX1 \
      --elimination-bed data/AG_v50_eliminated_strict.bed \
      --mapping-bed data/as_v50_prepde_to_postpde_coordinates.bed \
      --chrom-sizes data/AG_v50_chrom_sizes.txt \
      --pre-dir eigenvectors/prepde \
      --post-dir eigenvectors/postpde \
      --output-dir results/

In-script configuration:
  STAGES_TO_PLOT : list of stage prefixes to include (edit in script)
  FLIP_STAGES    : dict of {chromosome: [stages_to_flip]} for EV1 sign
                   correction (edit in script)
  PUB_SETTINGS   : dict of visual parameters (edit in script)

Input files:
  --elimination-bed: BED file of Ascaris eliminated DNA regions.
  --mapping-bed: Pre-PDE to post-PDE coordinate mapping (4-column:
      pre_chr  start  end  post_chr).  Maps post-PDE somatic chromosomes
      back to pre-PDE germline coordinates.
  --chrom-sizes: Chromosome sizes file (chr  length).
  --pre-dir / --post-dir: Directories containing FANC-format eigenvector
      files (chr  start  end  eigenvector).  The script searches for files
      matching each stage prefix in STAGES_TO_PLOT.
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

# ===== PLOT STYLING ===========================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'  # editable text in SVG for Illustrator

# ===== IN-SCRIPT CONFIGURATION ================================================
# Edit these per analysis run.  Stages are searched as glob prefixes in the
# --pre-dir and --post-dir directories.

STAGES_TO_PLOT = [
    'as_teste_iced_100kb',
    'as_ovary_iced_100kb',
    'as_0hr_iced_100kb',
    'as_48hr_iced_100kb',
    'as_60hr_iced_100kb',
    'as_5day_iced_100kb',
    'as_10day_iced_100kb',
]

# Per-chromosome EV1 sign correction.  EV1 sign is arbitrary in PCA; flip
# specific stages so compartment A/B direction is consistent across
# developmental stages for visual comparison.
# Format: {chromosome: [list_of_stage_prefixes_to_flip]}
FLIP_STAGES = {
    "chr01": ["as_ovary_iced_100kb", "as_48hr_iced_100kb", "as_60hr_iced_100kb"],
    "chrX1": ["as_teste_iced_100kb", "as_60hr_iced_100kb"],
}

# Visual parameters for publication-quality figures.
PUB_SETTINGS = {
    'line_width': 3.0,
    'eliminated_alpha': 0.4,
    'font_size_title': 18,
    'font_size_labels': 16,
    'font_size_stage': 14,
    'font_size_segments': 12,
    'grid_alpha': 0.3,
    'figure_width': 14,
    'subplot_height': 2.5,
}

# ==============================================================================
#  I/O helpers
# ==============================================================================

def load_eliminated_regions(bed_file):
    """Load eliminated regions from BED file.

    Returns list of (chrom, start, end) tuples.
    """
    regions = []
    with open(bed_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and re.match(r"^chr(\d+|X\d+)$", parts[0]):
                regions.append((parts[0], int(parts[1]), int(parts[2])))
    print(f"  Loaded {len(regions)} eliminated regions from {bed_file}")
    return regions


def load_coordinate_mapping(mapping_file):
    """Load pre-PDE to post-PDE coordinate mapping.

    Returns:
        mapping    : dict  post_chr -> (pre_chr, germ_start, germ_end)
        pre_to_segs: dict  pre_chr  -> [(post_chr, germ_start, germ_end), ...]
    """
    mapping = {}
    pre_to_segs = {}
    with open(mapping_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                pre_chr = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                post_chr = parts[3]
                mapping[post_chr] = (pre_chr, start, end)
                pre_to_segs.setdefault(pre_chr, []).append((post_chr, start, end))
    print(f"  Loaded {len(mapping)} coordinate mappings from {mapping_file}")
    return mapping, pre_to_segs


def load_chrom_sizes(sizes_file):
    """Load chromosome sizes from tab-delimited file (chr  length)."""
    chr_lengths = {}
    with open(sizes_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and re.match(r"^chr(\d+|X\d+)$", parts[0]):
                chr_lengths[parts[0]] = int(parts[1])
    print(f"  Loaded sizes for {len(chr_lengths)} chromosomes from {sizes_file}")
    return chr_lengths


def load_eigenvector_data(filepath, target_chr, coord_mapping):
    """Load EV1 data for a target chromosome from a FANC eigenvector file.

    For post-PDE files, uses coord_mapping to translate somatic chromosome
    coordinates back to germline positions.

    Returns numpy array of shape (N, 2) with columns [position, eigenvector].
    """
    data = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                chrom = parts[0]
                start = int(parts[1])
                value = float(parts[3])

                if chrom == target_chr:
                    data.append((start, value))
                elif chrom in coord_mapping and coord_mapping[chrom][0] == target_chr:
                    offset = coord_mapping[chrom][1]
                    data.append((offset + start, value))

    return np.array(data) if data else np.array([]).reshape(0, 2)


def is_eliminated(pos, chrom, eliminated_regions):
    """Check if a genomic position falls within an eliminated region."""
    for e_chr, e_start, e_end in eliminated_regions:
        if e_chr == chrom and e_start <= pos <= e_end:
            return True
    return False


def apply_flipping(data, sample_name, target_chromosome):
    """Flip EV1 sign if this stage/chromosome combo is in FLIP_STAGES."""
    if target_chromosome in FLIP_STAGES and sample_name in FLIP_STAGES[target_chromosome]:
        data_copy = data.copy()
        data_copy[:, 1] = -data_copy[:, 1]
        return data_copy
    return data


def calculate_symmetric_ylim(all_values):
    """Calculate y-axis limits centered at 0 with 15% padding."""
    if len(all_values) == 0:
        return (-1.0, 1.0)
    max_abs = np.max(np.abs(all_values))
    limit = max_abs * 1.15
    return (-limit, limit)


# ==============================================================================
#  Plotting
# ==============================================================================

def plot_chromosome(target_chr, chr_length, stage_data, eliminated_regions,
                    pre_to_segs, output_dir):
    """Create stacked EV1 plot for one chromosome across all stages."""
    # Pre-calculate symmetric y-limits across all stages
    all_y = []
    processed = []
    for sample_name, data, is_pre in stage_data:
        if len(data) > 0:
            flipped = apply_flipping(data, sample_name, target_chr)
            processed.append((sample_name, flipped, is_pre))
            all_y.extend(flipped[:, 1])
        else:
            processed.append((sample_name, data, is_pre))

    y_min, y_max = calculate_symmetric_ylim(all_y)

    n_stages = len(processed)
    fig, axes = plt.subplots(
        n_stages, 1,
        figsize=(PUB_SETTINGS['figure_width'],
                 PUB_SETTINGS['subplot_height'] * n_stages),
        sharex=True)
    if n_stages == 1:
        axes = [axes]

    line_color = '#E91E63' if target_chr.startswith('chrX') else '#2196F3'

    for idx, (sample_name, data, is_pre) in enumerate(processed):
        ax = axes[idx]

        if len(data) == 0:
            ax.text(0.5, 0.5, f"No data for {sample_name}",
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=PUB_SETTINGS['font_size_labels'], alpha=0.6)
            ax.set_ylim(y_min, y_max)
            continue

        if is_pre:
            ax.plot(data[:, 0], data[:, 1], color=line_color,
                    linewidth=PUB_SETTINGS['line_width'], alpha=0.9)
        else:
            # Post-PDE: break traces at eliminated regions
            seg_x, seg_y = [], []
            for pos, val in data:
                if is_eliminated(pos, target_chr, eliminated_regions):
                    if seg_x:
                        ax.plot(seg_x, seg_y, color=line_color,
                                linewidth=PUB_SETTINGS['line_width'], alpha=0.9)
                        seg_x, seg_y = [], []
                else:
                    seg_x.append(pos)
                    seg_y.append(val)
            if seg_x:
                ax.plot(seg_x, seg_y, color=line_color,
                        linewidth=PUB_SETTINGS['line_width'], alpha=0.9)

        # Shade eliminated regions
        for e_chr, e_start, e_end in eliminated_regions:
            if e_chr == target_chr:
                ax.axvspan(e_start, e_end, color='red',
                           alpha=PUB_SETTINGS['eliminated_alpha'],
                           linewidth=0, zorder=0)

        ax.axhline(0, linestyle=':', color='gray', alpha=0.5, linewidth=1.5,
                   zorder=1)
        ax.set_xlim(0, chr_length)
        ax.set_ylim(y_min, y_max)

        # Stage label on y-axis
        stage_label = sample_name.replace('as_', '').replace('_iced_100kb', '')
        if (target_chr in FLIP_STAGES and
                sample_name in FLIP_STAGES[target_chr]):
            stage_label += " (flipped)"
        ax.set_ylabel(stage_label, fontsize=PUB_SETTINGS['font_size_stage'],
                      fontweight='bold', rotation=0, ha='right', va='center')

        # Post-PDE segment labels on first panel for X chromosomes
        if (idx == 0 and target_chr in pre_to_segs and
                len(pre_to_segs[target_chr]) > 1 and
                target_chr.startswith('chrX')):
            y_range = y_max - y_min
            for post_chr, offset, end_pos in pre_to_segs[target_chr]:
                mid = offset + (end_pos - offset) / 2
                ax.text(mid, y_max + y_range * 0.08, post_chr,
                        ha='center', va='bottom',
                        fontsize=PUB_SETTINGS['font_size_segments'],
                        fontweight='bold', color='darkblue')

        ax.grid(True, alpha=PUB_SETTINGS['grid_alpha'], axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=PUB_SETTINGS['font_size_stage'])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.1f}"))

    axes[-1].set_xlabel(f'{target_chr} Position (Mb)', fontweight='bold',
                        fontsize=PUB_SETTINGS['font_size_labels'])

    if target_chr.startswith('chrX'):
        title_chr = f"X chromosome ({target_chr})"
    else:
        chr_num = re.search(r"\d+", target_chr)
        title_chr = f"Chromosome {chr_num.group()}" if chr_num else target_chr

    fig.suptitle(f"Eigenvector 1 across development - {title_chr}",
                 fontsize=PUB_SETTINGS['font_size_title'], fontweight='bold',
                 y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.4)

    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, f"eigen_{target_chr}_stacked")
    fig.savefig(f"{prefix}.png", dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(f"{prefix}.svg", bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {prefix}.png, {prefix}.svg")


# ==============================================================================
#  Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Stacked EV1 plot across Ascaris developmental stages')
    parser.add_argument('--chromosomes', nargs='+', default=['chr01', 'chrX1'],
                        help='Chromosomes to plot (default: chr01 chrX1)')
    parser.add_argument('--elimination-bed', required=True,
                        help='BED file with eliminated DNA regions')
    parser.add_argument('--mapping-bed', required=True,
                        help='Pre-PDE to post-PDE coordinate mapping file')
    parser.add_argument('--chrom-sizes', required=True,
                        help='Chromosome sizes file (chr  length)')
    parser.add_argument('--pre-dir', default='data/eigenvectors/ascaris/prepde',
                        help='Directory with pre-PDE eigenvector files')
    parser.add_argument('--post-dir', default='data/eigenvectors/ascaris/postpde',
                        help='Directory with post-PDE eigenvector files')
    parser.add_argument('--output-dir', default='results',
                        help='Output directory for figures')
    args = parser.parse_args()

    print("=" * 60)
    print("Ascaris EV1 Stacked Developmental Plot")
    print("=" * 60)

    # Load reference data
    eliminated = load_eliminated_regions(args.elimination_bed)
    coord_mapping, pre_to_segs = load_coordinate_mapping(args.mapping_bed)
    chr_lengths = load_chrom_sizes(args.chrom_sizes)

    # Validate chromosomes
    for chrom in args.chromosomes:
        if chrom not in chr_lengths:
            print(f"ERROR: {chrom} not in chromosome sizes file.")
            print(f"Available: {sorted(chr_lengths.keys())}")
            sys.exit(1)

    # Find eigenvector files for each stage
    print(f"\nSearching for eigenvector files...")
    stage_files = []
    for stage in STAGES_TO_PLOT:
        pre_hits = glob.glob(os.path.join(args.pre_dir,
                                          f"{stage}*.matrix.eigenvector"))
        post_hits = glob.glob(os.path.join(args.post_dir,
                                           f"{stage}*.matrix.eigenvector"))
        if pre_hits:
            stage_files.append((stage, pre_hits[0], True))
            print(f"  {stage}: pre-PDE")
        elif post_hits:
            stage_files.append((stage, post_hits[0], False))
            print(f"  {stage}: post-PDE")
        else:
            print(f"  {stage}: WARNING - not found")

    if not stage_files:
        print("ERROR: No eigenvector files found.")
        sys.exit(1)

    # Plot each chromosome
    for chrom in args.chromosomes:
        print(f"\nPlotting {chrom} ({chr_lengths[chrom] / 1e6:.1f} Mb)...")
        if chrom in FLIP_STAGES:
            print(f"  Flipped stages: {FLIP_STAGES[chrom]}")

        # Load data for all stages
        stage_data = []
        for stage, filepath, is_pre in stage_files:
            data = load_eigenvector_data(filepath, chrom, coord_mapping)
            stage_data.append((stage, data, is_pre))

        plot_chromosome(chrom, chr_lengths[chrom], stage_data, eliminated,
                        pre_to_segs, args.output_dir)

    print(f"\nDone. {len(args.chromosomes) * 2} files created in {args.output_dir}/")


if __name__ == '__main__':
    main()
