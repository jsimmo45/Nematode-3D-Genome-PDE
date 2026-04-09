#!/usr/bin/env python3
"""
distance_decay.py
=================
Hi-C distance decay analysis pipeline for Ascaris.  Computes, combines,
normalizes, and plots expected contact frequency as a function of genomic
distance across developmental timepoints.

Three subcommands allow entering the pipeline at any stage:

  run     — Full pipeline: run FAN-C `fanc expected` on .hic files, then
            combine, bin, normalize, and plot.  Requires FAN-C installed.
  combine — Start from existing FAN-C output text files: combine samples,
            re-bin at specified resolution(s), normalize, and plot.
  plot    — Final step only: read a pre-computed normalized text file and
            produce a publication-quality log-log distance decay plot.

Normalization scales each sample so that its expected contact value in the
first distance bin equals the average first-bin value across all samples.

Outputs:
  - Combined and normalized data tables (tab-delimited text)
  - Log-log distance decay plots (SVG, PNG, PDF)
  - Per-chromosome combined plots (run/combine modes)

Dependencies:
  pandas, matplotlib
  FAN-C (https://fan-c.readthedocs.io/) — only needed for 'run' subcommand

Example usage:
  # Full pipeline from .hic files (requires FAN-C):
  python distance_decay.py run \\
      --chrom-sizes data/AG_v50_chrom_sizes.txt \\
      --samples AG.v50.1cell.iced.5kb.hic AG.v50.48hr.iced.5kb.hic \\
      --output-binning 100000 \\
      --outdir results/

  # Combine existing FAN-C output files:
  python distance_decay.py combine \\
      --pattern 'expected/AG.v50.*.iced.5kb/*genome.txt' \\
      --output-binning 100000 \\
      --outdir results/

  # Plot from pre-computed normalized file:
  python distance_decay.py plot \\
      --input data/combined_whole_genome_100000bp_normalized.txt \\
      --output distance_decay \\
      --title 'Normalized Whole Genome Expected Contacts'
"""

import os
import sys
import glob
import argparse
import subprocess
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

# Publication-quality settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'


# ===========================================================================
# Shared helpers
# ===========================================================================

def normalize_binned_df(df, sample_cols):
    """Normalize so each sample's first-bin value equals the cross-sample average."""
    df_sorted = df.sort_values('binned_distance').reset_index(drop=True)
    first_vals = {s: df_sorted[s].iloc[0] for s in sample_cols}
    avg_first = sum(first_vals.values()) / len(first_vals)
    df_norm = df_sorted.copy()
    for s in sample_cols:
        if first_vals[s] != 0:
            df_norm[s] = df_norm[s] * (avg_first / first_vals[s])
    return df_norm


def clean_sample_name(name):
    """Remove common prefixes/suffixes for cleaner legend labels."""
    return name.replace('AG.v50.', '').replace('.iced.5kb', '')


def find_common_prefix(strings):
    """Return longest common prefix of a list of strings."""
    if not strings:
        return ""
    shortest = min(strings, key=len)
    for i, ch in enumerate(shortest):
        for other in strings:
            if other[i] != ch:
                return shortest[:i]
    return shortest


# ===========================================================================
# 'run' subcommand — full pipeline via FAN-C
# ===========================================================================

def get_chromosomes(chrom_sizes_path):
    """Parse chrom.sizes, excluding chrM/chrUn/random/alt."""
    chroms = []
    with open(chrom_sizes_path) as f:
        for line in f:
            if not line.strip():
                continue
            name = line.strip().split()[0]
            if any(x in name for x in ['chrM', 'Un', 'random', 'alt']):
                continue
            chroms.append(name)
    return chroms


def run_fanc_expected(hic_file, outdir, chromosomes):
    """Run 'fanc expected' on whole genome and each chromosome."""
    sample_name = os.path.splitext(os.path.basename(hic_file))[0]
    sample_dir = os.path.join(outdir, sample_name)
    os.makedirs(sample_dir, exist_ok=True)

    # Whole genome
    genome_txt = os.path.join(sample_dir, f'{sample_name}_genome.txt')
    genome_png = os.path.join(sample_dir, f'{sample_name}_genome_fanc.png')
    subprocess.run(['fanc', 'expected', '-p', genome_png, hic_file, genome_txt],
                   check=True)

    txt_files = {'genome': genome_txt}
    for chrom in chromosomes:
        chrom_txt = os.path.join(sample_dir, f'{sample_name}_{chrom}.txt')
        chrom_png = os.path.join(sample_dir, f'{sample_name}_{chrom}_fanc.png')
        subprocess.run(['fanc', 'expected', '-c', chrom, '-p', chrom_png,
                        hic_file, chrom_txt], check=True)
        txt_files[chrom] = chrom_txt

    return txt_files


def combine_and_plot_whole_genome(sample_to_txt, outdir, bin_sizes):
    """Merge whole-genome expected files, bin, normalize, and plot."""
    combined_dir = os.path.join(outdir, 'combined_whole_genome')
    normalized_dir = os.path.join(outdir, 'normalized_whole_genome')
    os.makedirs(combined_dir, exist_ok=True)
    os.makedirs(normalized_dir, exist_ok=True)

    # Merge samples
    merged_df = None
    sample_names = []
    for sample_name, txt_map in sample_to_txt.items():
        df = pd.read_csv(txt_map['genome'], sep='\t', skiprows=1, header=None,
                         names=['distance', sample_name])
        sample_names.append(sample_name)
        merged_df = df if merged_df is None else pd.merge(
            merged_df, df, on='distance', how='inner')

    # Save unbinned merged table
    merged_df.to_csv(os.path.join(combined_dir, 'combined_whole_genome.txt'),
                     sep='\t', index=False)

    for bin_size in bin_sizes:
        df_binned = merged_df.copy()
        df_binned['binned_distance'] = (df_binned['distance'] // bin_size) * bin_size
        df_grouped = df_binned.groupby('binned_distance').mean().reset_index()

        # Save binned table
        df_grouped.to_csv(
            os.path.join(combined_dir,
                         f'combined_whole_genome_{bin_size}bp.txt'),
            sep='\t', index=False)

        # Unnormalized plot
        _plot_decay(df_grouped, sample_names, 'binned_distance',
                    f'Combined Whole Genome (bin: {bin_size} bp)',
                    'Expected contact',
                    os.path.join(combined_dir,
                                 f'combined_whole_genome_{bin_size}bp'))

        # Normalized
        df_norm = normalize_binned_df(df_grouped, sample_names)
        norm_path = os.path.join(normalized_dir,
                                 f'combined_whole_genome_{bin_size}bp_normalized')
        df_norm.to_csv(f'{norm_path}.txt', sep='\t', index=False)
        _plot_decay(df_norm, sample_names, 'binned_distance',
                    f'Combined Whole Genome (Normalized, bin: {bin_size} bp)',
                    'Normalized expected contact', norm_path)

    print(f"  Whole-genome plots saved to {combined_dir}/ and {normalized_dir}/")


def combine_and_plot_chromosomes(sample_to_txt, chromosomes, outdir,
                                 bin_size=100000):
    """Produce per-chromosome combined normalized plots."""
    chrom_dir = os.path.join(outdir, 'chromosomes_normalized')
    os.makedirs(chrom_dir, exist_ok=True)

    for chrom in chromosomes:
        merged_df = None
        sample_names = []
        for sample_name, txt_map in sample_to_txt.items():
            if chrom not in txt_map:
                continue
            df = pd.read_csv(txt_map[chrom], sep='\t', skiprows=1, header=None,
                             names=['distance', sample_name])
            sample_names.append(sample_name)
            merged_df = df if merged_df is None else pd.merge(
                merged_df, df, on='distance', how='inner')

        if merged_df is None or not sample_names:
            continue

        merged_df['binned_distance'] = (merged_df['distance'] // bin_size) * bin_size
        df_grouped = merged_df.groupby('binned_distance').mean().reset_index()
        df_norm = normalize_binned_df(df_grouped, sample_names)

        prefix = os.path.join(chrom_dir, f'{chrom}_normalized')
        df_norm.to_csv(f'{prefix}.txt', sep='\t', index=False)
        _plot_decay(df_norm, sample_names, 'binned_distance',
                    f'{chrom} Normalized (100 kb bins)',
                    'Normalized expected contact', prefix,
                    figsize=(5, 4), dpi=150)


def cmd_run(args):
    """Execute the full FAN-C pipeline."""
    os.makedirs(args.outdir, exist_ok=True)
    chromosomes = get_chromosomes(args.chrom_sizes)
    print(f"  Chromosomes: {len(chromosomes)}")

    # Run FAN-C for each sample
    sample_to_txt = {}
    for hic_file in args.samples:
        sample_name = os.path.splitext(os.path.basename(hic_file))[0]
        print(f"\n  Running fanc expected for {sample_name}...")
        txt_map = run_fanc_expected(hic_file, args.outdir, chromosomes)
        sample_to_txt[sample_name] = txt_map

    # Combine, normalize, plot
    print("\n  Combining whole-genome data...")
    combine_and_plot_whole_genome(sample_to_txt, args.outdir,
                                 args.output_binning)

    print("\n  Combining per-chromosome data...")
    combine_and_plot_chromosomes(sample_to_txt, chromosomes, args.outdir)

    print("\nDone.")


# ===========================================================================
# 'combine' subcommand — from existing FAN-C text files
# ===========================================================================

def cmd_combine(args):
    """Combine existing FAN-C output files."""
    os.makedirs(args.outdir, exist_ok=True)
    norm_dir = os.path.join(args.outdir, 'normalized')
    os.makedirs(norm_dir, exist_ok=True)

    txt_files = sorted(glob.glob(args.pattern))
    if not txt_files:
        print(f"No files found matching: {args.pattern}")
        sys.exit(1)
    print(f"  Found {len(txt_files)} files")

    # Build sample names from filenames
    full_names = [os.path.basename(f).replace('_genome.txt', '') for f in txt_files]
    prefix = find_common_prefix(full_names)

    # Read and merge
    df_list = []
    for txt_file, full_name in zip(txt_files, full_names):
        short = full_name[len(prefix):].lstrip('._-')
        df = pd.read_csv(txt_file, sep='\t', skiprows=1, header=None,
                         names=['distance', short])
        df_list.append(df)

    merged = df_list[0]
    for df in df_list[1:]:
        merged = pd.merge(merged, df, on='distance', how='inner')

    sample_cols = [c for c in merged.columns if c != 'distance']

    # Bin, normalize, plot for each bin size
    for bin_size in args.output_binning:
        df_binned = merged.copy()
        df_binned['binned_distance'] = (df_binned['distance'] // bin_size) * bin_size
        df_grouped = df_binned.groupby('binned_distance').mean().reset_index()

        # Save and plot unnormalized
        prefix_path = os.path.join(args.outdir, f'combined_{bin_size}bp')
        df_grouped.to_csv(f'{prefix_path}.txt', sep='\t', index=False)
        _plot_decay(df_grouped, sample_cols, 'binned_distance',
                    f'Combined Whole Genome (bin: {bin_size} bp)',
                    'Expected contact', prefix_path)

        # Normalize, save, plot
        df_norm = normalize_binned_df(df_grouped, sample_cols)
        norm_path = os.path.join(norm_dir, f'combined_{bin_size}bp_normalized')
        df_norm.to_csv(f'{norm_path}.txt', sep='\t', index=False)
        _plot_decay(df_norm, sample_cols, 'binned_distance',
                    f'Combined (Normalized, bin: {bin_size} bp)',
                    'Normalized expected contact', norm_path)

    print(f"\nDone. Results in {args.outdir}/")


# ===========================================================================
# 'plot' subcommand — plot from final normalized file
# ===========================================================================

def cmd_plot(args):
    """Plot a pre-computed normalized distance decay file."""
    df = pd.read_csv(args.input, sep='\t')

    # Find distance column
    dist_col = 'binned_distance'
    if dist_col not in df.columns:
        dist_cols = [c for c in df.columns if 'distance' in c.lower()]
        if not dist_cols:
            print("ERROR: No distance column found")
            sys.exit(1)
        dist_col = dist_cols[0]

    sample_cols = [c for c in df.columns if c != dist_col]
    if not sample_cols:
        print("ERROR: No sample columns found")
        sys.exit(1)

    print(f"  Distance column: {dist_col}")
    print(f"  Samples: {sample_cols}")

    # Create output directory if needed
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    _plot_decay(df, sample_cols, dist_col, args.title,
                'Normalized expected contact', args.output,
                figsize=(args.width, args.height), dpi=args.dpi,
                clean_names=True)

    print(f"\nSaved: {args.output}.svg/.png/.pdf")


# ===========================================================================
# Shared plotting
# ===========================================================================

def _plot_decay(df, sample_cols, dist_col, title, ylabel, output_prefix,
                figsize=(8, 6), dpi=300, clean_names=False):
    """Generate a log-log distance decay plot and save as SVG/PNG/PDF."""
    plt.figure(figsize=figsize)

    n = len(sample_cols)
    colors = cm.viridis([i / max(1, n - 1) for i in range(n)]) if n > 1 else ['blue']

    for i, sample in enumerate(sample_cols):
        mask = ~(pd.isna(df[dist_col]) | pd.isna(df[sample]))
        x = df.loc[mask, dist_col]
        y = df.loc[mask, sample]
        label = clean_sample_name(sample) if clean_names else sample
        plt.plot(x, y, label=label, color=colors[i], linewidth=0.75)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Binned genomic distance (bp)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()

    for ext in ['svg', 'png', 'pdf']:
        plt.savefig(f'{output_prefix}.{ext}', format=ext, dpi=dpi,
                    bbox_inches='tight')
    plt.close()


# ===========================================================================
# Main — subcommand dispatch
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Hi-C distance decay pipeline: run FAN-C, combine, '
                    'normalize, and plot')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- run ---
    p_run = subparsers.add_parser(
        'run', help='Full pipeline: run FAN-C, combine, normalize, plot')
    p_run.add_argument('--chrom-sizes', required=True,
                       help='Chromosome sizes file (chr  length)')
    p_run.add_argument('--samples', nargs='+', required=True,
                       help='.hic files to process')
    p_run.add_argument('--output-binning', type=int, nargs='+', required=True,
                       help='Bin sizes in bp (e.g., 100000)')
    p_run.add_argument('--outdir', default='expected_combined',
                       help='Output directory')

    # --- combine ---
    p_comb = subparsers.add_parser(
        'combine', help='Combine existing FAN-C output files')
    p_comb.add_argument('--pattern', required=True,
                        help='Glob pattern for FAN-C genome.txt files')
    p_comb.add_argument('--output-binning', type=int, nargs='+', required=True,
                        help='Bin sizes in bp')
    p_comb.add_argument('--outdir', default='combined_output',
                        help='Output directory')

    # --- plot ---
    p_plot = subparsers.add_parser(
        'plot', help='Plot from pre-computed normalized file')
    p_plot.add_argument('--input', required=True,
                        help='Normalized distance decay text file')
    p_plot.add_argument('--output', default='distance_decay',
                        help='Output filename prefix')
    p_plot.add_argument('--title', default='Normalized Whole Genome Expected Contacts',
                        help='Plot title')
    p_plot.add_argument('--dpi', type=int, default=300)
    p_plot.add_argument('--width', type=float, default=8,
                        help='Figure width in inches')
    p_plot.add_argument('--height', type=float, default=6,
                        help='Figure height in inches')

    args = parser.parse_args()

    print("=" * 60)
    print(f"Distance Decay Analysis — {args.command}")
    print("=" * 60)

    if args.command == 'run':
        cmd_run(args)
    elif args.command == 'combine':
        cmd_combine(args)
    elif args.command == 'plot':
        cmd_plot(args)


if __name__ == '__main__':
    main()
