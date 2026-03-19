#!/usr/bin/env python3
"""
multiomics_ab_compartments.py
=============================
Compare multi-omics signal levels between A and B chromatin compartments in
Ascaris.  For each omics dataset (ChIP-seq, ATAC-seq, RNA-seq, etc.),
genomic bins are classified as A or B based on Hi-C eigenvector sign, and
signal distributions are compared using a Mann-Whitney U test.

Produces per-dataset box-and-whisker plots with jittered data points
showing signal in A vs B compartments, plus a summary statistics CSV.

Outputs (per dataset):
  - Box plot with jittered points (PNG + SVG)
  - Summary statistics CSV across all datasets

Dependencies:
  numpy, pandas, matplotlib, scipy

Example usage:
  python multiomics_ab_compartments.py \\
      --eigenvector data/eigenvectors/ascaris/postpde/as_5day_iced_100kb.matrix.eigenvector \\
      --base-dir omics_data/ \\
      --ev-threshold 0.15 \\
      --output-dir results/

In-script configuration:
  DATASETS       : dict defining each of the 8 omics datasets (name, color,
                   pattern, subdirectory under --base-dir, y-axis max)
  SIGNAL_FILTERS : dict of per-dataset minimum signal thresholds (bins below
                   threshold are excluded; set to 0.0 to include all data)

Input files:
  --eigenvector: FANC-format eigenvector file (chr  start  end  PC1) at
      100 kb resolution.  Bins with positive PC1 = A compartment, negative = B.
  Omics bedgraph files: 10 kb resolution signal tracks (chr  start  end  value)
      organized in subdirectories under --base-dir.  The script averages
      10 kb bins falling within each 100 kb eigenvector bin.
  Optional: pre_to_post.bed in --base-dir for post-PDE → pre-PDE coordinate
      mapping (auto-detected; analysis proceeds without it).
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 1.0,
    'svg.fonttype': 'none',   # editable text in SVG for Illustrator
})

# ===========================================================================
# IN-SCRIPT CONFIGURATION — edit these per analysis run
# ===========================================================================

# Dataset definitions.  Each entry maps a short key to:
#   name          : display name for axis labels and titles
#   color         : hex color for plot elements
#   file_pattern  : glob pattern for bedgraph files (replicates averaged)
#   directory     : subdirectory under --base-dir
#   ymax          : y-axis maximum (None = auto-scale)
DATASETS = {
    'rnaseq': {
        'name': 'RNA-seq',
        'color': '#E74C3C',
        'file_pattern': 'rna_seq_5day.norm.mean0.10kb.bg',
        'directory': 'rnaseq',
        'ymax': 40,
    },
    'atacseq': {
        'name': 'ATAC-seq',
        'color': '#2ECC71',
        'file_pattern': 'atac_5day_*.mean0.10kb.bg',
        'directory': 'atacseq',
        'ymax': 10,
    },
    'h3k4me3': {
        'name': 'H3K4me3',
        'color': '#9B59B6',
        'file_pattern': 'H3K4me3_5day_*.mean0.10kb.bg',
        'directory': 'histones/h3k4me3',
        'ymax': 35,
    },
    'h3s10p': {
        'name': 'H3S10p',
        'color': '#4CAF50',
        'file_pattern': 'H3S10p_5day_*.mean0.10kb.bg',
        'directory': 'histones/h3s10p',
        'ymax': 20,
    },
    'h3k36me2': {
        'name': 'H3K36me2',
        'color': '#795548',
        'file_pattern': 'H3K36me2_5day_*.mean0.10kb.bg',
        'directory': 'histones/h3k36me2',
        'ymax': 40,
    },
    'h3k36me3': {
        'name': 'H3K36me3',
        'color': '#E91E63',
        'file_pattern': 'H3K36me3_5day_*.mean0.10kb.bg',
        'directory': 'histones/h3k36me3',
        'ymax': 45,
    },
    'h3k9me3': {
        'name': 'H3K9me3',
        'color': '#1ABC9C',
        'file_pattern': 'H3K9me3_5day_*.mean0.10kb.bg',
        'directory': 'histones/h3k9me3',
        'ymax': 25,
    },
    'pol2': {
        'name': 'Pol II',
        'color': '#3F51B5',
        'file_pattern': 'Pol2_5day_*.mean0.10kb.bg',
        'directory': 'histones/pol2',
        'ymax': 50,
    },
}

# Per-dataset minimum signal thresholds.  Bins with signal below this value
# are excluded.  Set to 0.0 to include all data (no filtering).
SIGNAL_FILTERS = {k: 0.0 for k in DATASETS}


# ===========================================================================
# I/O helpers
# ===========================================================================

def load_eigenvectors(ev_file, threshold=0.0):
    """Load eigenvector file and filter weak compartment calls.

    Parameters
    ----------
    ev_file : Path
        FANC-format eigenvector file (chr  start  end  PC1).
    threshold : float
        Minimum |eigenvector| to retain a bin.

    Returns
    -------
    pd.DataFrame with columns [chr, start, end, eigenvector].
    """
    ev_df = pd.read_csv(ev_file, sep='\t', header=None,
                        names=['chr', 'start', 'end', 'eigenvector'])
    ev_df = ev_df.dropna()

    n_before = len(ev_df)
    ev_df = ev_df[abs(ev_df['eigenvector']) >= threshold]
    n_after = len(ev_df)

    print(f"  Loaded {n_after} bins (filtered {n_before - n_after} "
          f"with |EV1| < {threshold})")
    return ev_df


def load_genome_conversion(base_dir):
    """Load optional post-PDE → pre-PDE coordinate mapping.

    Looks for pre_to_post.bed in base_dir.  Returns DataFrame or None.
    """
    conv_file = base_dir / 'pre_to_post.bed'
    if not conv_file.exists():
        print("  No pre_to_post.bed found — proceeding without coordinate "
              "conversion.")
        return None

    conv_df = pd.read_csv(conv_file, sep='\t', header=None,
                          names=['pre_chr', 'pre_start', 'pre_end', 'post_chr'])
    print(f"  Loaded genome conversion: {len(conv_df)} regions")
    return conv_df


def map_eigenvectors_to_prepde(ev_df, genome_conversion):
    """Map post-PDE eigenvector coordinates to pre-PDE for matching with
    omics data that is on the pre-PDE assembly."""
    if genome_conversion is None:
        return ev_df

    print("  Mapping eigenvectors to pre-PDE coordinates...")
    mapped = []
    for _, ev_bin in ev_df.iterrows():
        matches = genome_conversion[
            genome_conversion['post_chr'] == ev_bin['chr']]
        if len(matches) > 0:
            mapped.append({
                'chr': matches.iloc[0]['pre_chr'],
                'start': ev_bin['start'],
                'end': ev_bin['end'],
                'eigenvector': ev_bin['eigenvector'],
            })

    result = pd.DataFrame(mapped)
    print(f"  Mapped {len(result)} bins")
    return result


def load_omics_data(dataset_key, base_dir):
    """Load and optionally average replicate bedgraph files for one dataset.

    Applies the signal filter defined in SIGNAL_FILTERS.

    Returns pd.DataFrame with columns [chr, start, end, value] or None.
    """
    info = DATASETS[dataset_key]
    data_dir = base_dir / info['directory']
    pattern = info['file_pattern']

    # Find files
    if '*' in pattern:
        files = list(data_dir.glob(pattern))
    else:
        fp = data_dir / pattern
        files = [fp] if fp.exists() else []

    if not files:
        print(f"  No files found for {info['name']} in {data_dir}")
        return None

    # Load and average replicates
    dfs = []
    for f in files:
        df = pd.read_csv(f, sep='\t', header=None,
                         names=['chr', 'start', 'end', 'value'])
        dfs.append(df)
        print(f"    Loaded {f.name}: {len(df)} bins")

    if len(dfs) == 1:
        result = dfs[0]
    else:
        print(f"  Averaging {len(dfs)} replicates...")
        result = dfs[0].copy()
        for i, df in enumerate(dfs[1:], 1):
            result = result.merge(df, on=['chr', 'start', 'end'],
                                  how='outer', suffixes=('', f'_{i}'))
        value_cols = [c for c in result.columns if c.startswith('value')]
        result['value'] = result[value_cols].mean(axis=1)
        result = result[['chr', 'start', 'end', 'value']]

    # Apply signal filter
    thresh = SIGNAL_FILTERS.get(dataset_key, 0.0)
    n_before = len(result)
    result = result[result['value'] >= thresh].dropna()
    print(f"  {len(result)} bins after signal filter (>= {thresh})")
    return result


def match_omics_to_eigenvectors(omics_df, ev_df):
    """Match 10 kb omics bins to 100 kb eigenvector bins.

    For each eigenvector bin, averages the signal from all overlapping
    omics bins and assigns A/B compartment based on eigenvector sign.

    Returns pd.DataFrame with columns [chr, start, end, eigenvector,
    value, compartment, n_omics_bins].
    """
    matched = []
    for _, ev_bin in ev_df.iterrows():
        overlaps = omics_df[
            (omics_df['chr'] == ev_bin['chr']) &
            (omics_df['start'] < ev_bin['end']) &
            (omics_df['end'] > ev_bin['start'])]

        if len(overlaps) > 0:
            matched.append({
                'chr': ev_bin['chr'],
                'start': ev_bin['start'],
                'end': ev_bin['end'],
                'eigenvector': ev_bin['eigenvector'],
                'value': overlaps['value'].mean(),
                'compartment': 'A' if ev_bin['eigenvector'] > 0 else 'B',
                'n_omics_bins': len(overlaps),
            })

    result = pd.DataFrame(matched)
    if len(result) > 0:
        print(f"  Matched {len(result)} bins "
              f"(avg {result['n_omics_bins'].mean():.1f} omics bins per EV bin)")
    return result


# ===========================================================================
# Plotting
# ===========================================================================

def create_boxplot(dataset_key, matched_df, output_dir):
    """Create box-and-whisker plot with jittered data points for A vs B.

    Returns dict of summary statistics.
    """
    info = DATASETS[dataset_key]

    a_vals = matched_df[matched_df['compartment'] == 'A']['value'].values
    b_vals = matched_df[matched_df['compartment'] == 'B']['value'].values

    ymax = info.get('ymax') or np.max(np.concatenate([a_vals, b_vals])) * 1.1
    ylabel = f"{info['name']} signal"

    # --- Mann-Whitney U test ---
    stat, p_value = stats.mannwhitneyu(a_vals, b_vals, alternative='two-sided')

    if p_value < 0.001:
        sig = '***'
    elif p_value < 0.01:
        sig = '**'
    elif p_value < 0.05:
        sig = '*'
    else:
        sig = 'ns'

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(5, 5))

    # Box plot (outliers hidden — we draw points manually)
    bp = ax.boxplot([a_vals, b_vals], positions=[1, 2], widths=0.5,
                    patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor(info['color'])
        patch.set_alpha(0.7)

    # Add mean markers
    ax.scatter([1, 2], [np.mean(a_vals), np.mean(b_vals)],
               marker='D', color='red', s=40, zorder=5, label='Mean')

    # Jittered individual points (subsample if >1000)
    rng = np.random.default_rng(42)
    jitter = 0.1
    max_pts = 1000

    for pos, vals in [(1, a_vals), (2, b_vals)]:
        sample = vals if len(vals) <= max_pts else rng.choice(
            vals, max_pts, replace=False)

        normal = sample[sample <= ymax]
        clipped = sample[sample > ymax]

        # Normal points
        x = pos + rng.normal(0, jitter, len(normal))
        ax.scatter(x, normal, alpha=0.3, s=5, color=info['color'])

        # Clipped points shown as empty circles at y-max
        if len(clipped) > 0:
            x_clip = pos + rng.normal(0, jitter, len(clipped))
            ax.scatter(x_clip, np.full(len(clipped), ymax * 0.98),
                       alpha=0.6, s=8, facecolors='none',
                       edgecolors=info['color'], linewidths=0.8)

    # Clipped-point annotation
    n_clipped = sum(1 for v in np.concatenate([a_vals, b_vals]) if v > ymax)
    if n_clipped > 0:
        ax.text(0.98, 0.02,
                f'{n_clipped} points > y-max\n(shown as empty circles)',
                transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                fontsize=8)

    # Significance bracket
    y_sig = ymax * 0.92
    ax.plot([1, 2], [y_sig, y_sig], 'k-', linewidth=1)
    ax.text(1.5, y_sig * 1.02, sig, ha='center', fontsize=12,
            fontweight='bold')

    # Stats annotation
    ax.text(0.98, 0.98,
            f'A: n={len(a_vals)}, mean={np.mean(a_vals):.2f}\n'
            f'B: n={len(b_vals)}, mean={np.mean(b_vals):.2f}\n'
            f'p = {p_value:.2e}',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=9)

    # Legend for mean marker
    ax.legend(loc='upper left', framealpha=0.8,
              fontsize=9, markerscale=0.8,
              handletextpad=0.3)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['A', 'B'])
    ax.set_ylabel(ylabel)
    ax.set_title(f'{info["name"]} in A vs B Compartments')
    ax.set_ylim(0, ymax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    prefix = output_dir / f'{dataset_key}_AB_comparison'
    fig.savefig(f'{prefix}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{prefix}.svg', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {prefix}.png/.svg")

    return {
        'dataset': info['name'],
        'n_A': len(a_vals),
        'n_B': len(b_vals),
        'mean_A': np.mean(a_vals),
        'mean_B': np.mean(b_vals),
        'median_A': np.median(a_vals),
        'median_B': np.median(b_vals),
        'log2_fc': (np.log2(np.mean(a_vals) / np.mean(b_vals))
                    if np.mean(b_vals) > 0 else np.nan),
        'p_value': p_value,
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-omics A vs B compartment analysis for Ascaris')
    parser.add_argument('--eigenvector', required=True,
                        help='Eigenvector file (FANC format, 100 kb)')
    parser.add_argument('--base-dir', default='.',
                        help='Base directory containing omics subdirectories')
    parser.add_argument('--ev-threshold', type=float, default=0.15,
                        help='Minimum |eigenvector| to include a bin '
                             '(default: 0.15)')
    parser.add_argument('--output-dir', default='multiomics_AB_results',
                        help='Output directory')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Multi-omics A/B Compartment Analysis")
    print("=" * 60)

    # --- Load eigenvectors ---
    print(f"\nLoading eigenvectors from {args.eigenvector}...")
    ev_df = load_eigenvectors(Path(args.eigenvector), args.ev_threshold)

    # --- Optional coordinate conversion ---
    genome_conv = load_genome_conversion(base_dir)
    ev_df = map_eigenvectors_to_prepde(ev_df, genome_conv)

    # --- Check which datasets are available ---
    print("\nChecking data availability...")
    available = []
    for key in DATASETS:
        info = DATASETS[key]
        data_dir = base_dir / info['directory']
        pattern = info['file_pattern']
        if '*' in pattern:
            files = list(data_dir.glob(pattern))
        else:
            fp = data_dir / pattern
            files = [fp] if fp.exists() else []
        if files:
            available.append(key)
            print(f"  + {info['name']}: {len(files)} file(s)")
        else:
            print(f"  - {info['name']}: not found")

    if not available:
        print("ERROR: No datasets found. Check --base-dir and file patterns.")
        sys.exit(1)

    # --- Process each dataset ---
    all_stats = []
    for key in available:
        print(f"\n--- {DATASETS[key]['name']} ---")
        omics_df = load_omics_data(key, base_dir)
        if omics_df is None:
            continue

        matched = match_omics_to_eigenvectors(omics_df, ev_df)
        if len(matched) < 10:
            print(f"  Skipping — only {len(matched)} matched bins")
            continue

        result = create_boxplot(key, matched, output_dir)
        all_stats.append(result)

    # --- Save summary CSV ---
    if all_stats:
        csv_path = output_dir / 'AB_compartment_statistics.csv'
        pd.DataFrame(all_stats).to_csv(csv_path, index=False)
        print(f"\nSaved summary statistics: {csv_path}")

    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Results in {output_dir}/")
    for s in all_stats:
        direction = "A-enriched" if s['log2_fc'] > 0 else "B-enriched"
        sig = "***" if s['p_value'] < 0.001 else (
              "**" if s['p_value'] < 0.01 else (
              "*" if s['p_value'] < 0.05 else "ns"))
        print(f"  {s['dataset']}: {direction} (p={s['p_value']:.2e}) {sig}")


if __name__ == '__main__':
    main()
