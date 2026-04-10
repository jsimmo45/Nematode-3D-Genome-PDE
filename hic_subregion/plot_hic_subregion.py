#!/usr/bin/env python3
"""
Plot Hi-C contact heatmap for custom genomic coordinate ranges.

This script extracts and visualizes a specific region of a Hi-C contact matrix
based on user-defined genomic coordinates for both axes.

Usage:
    python plot_hic_subregion.py --bins bins.bed --matrix matrix.txt \\
        --x-coords "chr1:1000000-2000000,chr2:500000-1500000" \\
        --y-coords "chr3:0-1000000" \\
        --output my_plot \\
        --vmax 300

Author: Custom Hi-C visualization tool
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'  # editable text in SVG
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from pathlib import Path


# ============================================================================
# COLORMAP (same as main script)
# ============================================================================

def create_white_to_red_colormap():
    """
    Create custom colormap: white → yellow → orange → red → dark red → black.
    """
    colors = [
        '#FFFFFF',  # white (no contact)
        '#FFF7BC',  # very light yellow
        '#FEE391',  # light yellow
        '#FEC44F',  # yellow
        '#FE9929',  # yellow-orange
        '#EC7014',  # orange
        '#CC4C02',  # orange-red
        '#CC0000',  # red
        '#990000',  # dark red
        '#660000',  # very dark red
        '#330000',  # almost black
        '#000000'   # black (strongest contact)
    ]
    
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('white_to_red', colors, N=n_bins)
    cmap.set_bad(color='#CCCCCC')  # Grey for masked values
    return cmap


# ============================================================================
# COORDINATE PARSING
# ============================================================================

def parse_coordinates(coord_string):
    """
    Parse coordinate string into list of (chrom, start, end) tuples.
    
    Format: "chr1:1000000-2000000,chr2:500000-1500000"
    
    Returns:
        List of (chrom, start, end) tuples
    """
    regions = []
    
    for region_str in coord_string.split(','):
        region_str = region_str.strip()
        
        if ':' not in region_str or '-' not in region_str:
            raise ValueError(f"Invalid coordinate format: {region_str}. Expected chr:start-end")
        
        chrom, pos_range = region_str.split(':')
        start, end = pos_range.split('-')
        
        regions.append((chrom, int(start), int(end)))
    
    return regions


# ============================================================================
# DATA LOADING
# ============================================================================

def load_bins_file(bins_path):
    """Load HiC-Pro bins file."""
    print(f"Loading bins file: {bins_path}")
    
    df = pd.read_csv(
        bins_path,
        sep='\t',
        header=None,
        names=['chrom', 'start', 'end', 'bin_id']
    )
    
    print(f"  Loaded {len(df)} bins")
    return df


def load_matrix_file(matrix_path):
    """Load HiC-Pro sparse matrix file."""
    print(f"Loading matrix file: {matrix_path}")
    
    df = pd.read_csv(
        matrix_path,
        sep='\t',
        header=None,
        names=['bin_i', 'bin_j', 'count']
    )
    
    # Convert to 0-indexed
    df['bin_i'] -= 1
    df['bin_j'] -= 1
    
    print(f"  Loaded {len(df):,} interactions")
    return df


# ============================================================================
# BIN SELECTION
# ============================================================================

def select_bins_for_regions(bins_df, regions):
    """
    Select bins that overlap with the specified regions.
    
    Args:
        bins_df: DataFrame with bin annotations
        regions: List of (chrom, start, end) tuples
    
    Returns:
        DataFrame of selected bins, list of chromosome boundary indices
    """
    selected_bins = []
    chrom_boundaries = [0]  # Start of first chromosome
    
    for chrom, start, end in regions:
        # Find overlapping bins
        region_bins = bins_df[
            (bins_df['chrom'] == chrom) &
            (bins_df['start'] < end) &
            (bins_df['end'] > start)
        ].copy()
        
        if len(region_bins) == 0:
            print(f"  WARNING: No bins found for {chrom}:{start}-{end}")
            continue
        
        selected_bins.append(region_bins)
        chrom_boundaries.append(chrom_boundaries[-1] + len(region_bins))
        
        print(f"  Selected {len(region_bins)} bins for {chrom}:{start}-{end}")
    
    if not selected_bins:
        raise ValueError("No bins selected for the specified regions")
    
    # Combine all selected bins
    combined_bins = pd.concat(selected_bins, ignore_index=True)
    
    return combined_bins, chrom_boundaries[:-1]  # Don't include final boundary


def build_submatrix(matrix_df, x_bins, y_bins, cpm_normalize=True):
    """
    Extract submatrix for the selected bins.
    
    Args:
        matrix_df: Sparse matrix DataFrame
        x_bins: Bins for x-axis (columns)
        y_bins: Bins for y-axis (rows)
        cpm_normalize: Apply CPM normalization
    
    Returns:
        Dense contact matrix (y_bins × x_bins)
    """
    print("Building contact matrix...")
    
    # Create mapping from bin_id to index in submatrix
    x_bin_to_idx = {bin_id: idx for idx, bin_id in enumerate(x_bins['bin_id'].values)}
    y_bin_to_idx = {bin_id: idx for idx, bin_id in enumerate(y_bins['bin_id'].values)}
    
    # Get all bin IDs for efficient filtering
    x_bin_ids = set(x_bins['bin_id'].values)
    y_bin_ids = set(y_bins['bin_id'].values)
    
    # Initialize matrix
    contact_matrix = np.zeros((len(y_bins), len(x_bins)), dtype=np.float64)
    
    # Fill matrix - check both orientations
    n_used = 0
    for _, row in matrix_df.iterrows():
        bin_i = int(row['bin_i'])
        bin_j = int(row['bin_j'])
        count = row['count']
        
        # Check if bin_i is in y-axis and bin_j is in x-axis
        if bin_i in y_bin_ids and bin_j in x_bin_ids:
            if bin_i in y_bin_to_idx and bin_j in x_bin_to_idx:
                y_idx = y_bin_to_idx[bin_i]
                x_idx = x_bin_to_idx[bin_j]
                contact_matrix[y_idx, x_idx] += count
                n_used += 1
        
        # Check reverse: bin_j is in y-axis and bin_i is in x-axis
        if bin_j in y_bin_ids and bin_i in x_bin_ids:
            if bin_j in y_bin_to_idx and bin_i in x_bin_to_idx:
                y_idx = y_bin_to_idx[bin_j]
                x_idx = x_bin_to_idx[bin_i]
                contact_matrix[y_idx, x_idx] += count
                n_used += 1
    
    print(f"  Used {n_used:,} interactions")
    print(f"  Total contacts: {contact_matrix.sum():,.0f}")
    
    # CPM normalization
    if cpm_normalize:
        total_contacts = contact_matrix.sum()
        if total_contacts > 0:
            contact_matrix = (contact_matrix / total_contacts) * 1e6
            print(f"  Applied CPM normalization")
    
    # Get non-zero max for reference
    nonzero_vals = contact_matrix[contact_matrix > 0]
    if len(nonzero_vals) > 0:
        print(f"  Data range: {nonzero_vals.min():.2f} - {nonzero_vals.max():.2f}")
    
    return contact_matrix


# ============================================================================
# PLOTTING
# ============================================================================

def plot_heatmap(contact_matrix, x_bins, y_bins, x_regions, y_regions, 
                 x_boundaries, y_boundaries, output_path, vmax='20000', 
                 tick_interval=100000, dpi=300):
    """
    Plot contact heatmap with coordinate ticks and chromosome labels.
    
    Args:
        contact_matrix: Dense contact matrix (y × x)
        x_bins: Bins dataframe for x-axis
        y_bins: Bins dataframe for y-axis
        x_regions: List of (chrom, start, end) for x-axis
        y_regions: List of (chrom, start, end) for y-axis
        x_boundaries: Indices where x-axis chromosomes change
        y_boundaries: Indices where y-axis chromosomes change
        output_path: Output file prefix
        vmax: Maximum value for color scale (default: '20000')
        tick_interval: Interval for coordinate ticks in bp (default: 100000 = 100kb)
        dpi: Resolution for PNG
    """
    print(f"Generating heatmap...")
    
    # Determine vmax - default to 20000
    if vmax == 'auto':
        # Use actual maximum from the data
        nonzero = contact_matrix[contact_matrix > 0]
        if len(nonzero) > 0:
            vmax_val = np.max(nonzero)
        else:
            vmax_val = 20000.0
    elif isinstance(vmax, str) and vmax.startswith('p'):
        percentile = float(vmax[1:])
        vmax_val = np.nanpercentile(contact_matrix[~np.isnan(contact_matrix)], percentile)
    else:
        vmax_val = float(vmax)
    
    print(f"  Color scale vmax: {vmax_val:.2f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Get colormap
    cmap = create_white_to_red_colormap()
    
    # Plot heatmap
    im = ax.imshow(
        contact_matrix,
        cmap=cmap,
        aspect='auto',
        interpolation='none',
        vmin=0,
        vmax=vmax_val,
        origin='upper',
        extent=[0, contact_matrix.shape[1], contact_matrix.shape[0], 0]
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Contact Frequency (CPM)', fontsize=18, weight='bold')
    cbar.ax.tick_params(labelsize=14)
    
    # Add chromosome boundaries as dotted lines (at exact boundary indices)
    if len(x_boundaries) > 1:
        for boundary in x_boundaries[1:]:
            ax.axvline(x=boundary, color='black', linestyle='--', 
                      linewidth=1.5, alpha=0.7, zorder=10)
    
    if len(y_boundaries) > 1:
        for boundary in y_boundaries[1:]:
            ax.axhline(y=boundary, color='black', linestyle='--',
                      linewidth=1.5, alpha=0.7, zorder=10)
    
    # Generate coordinate ticks for x-axis
    x_tick_positions = []
    x_tick_labels = []
    
    for i, (chrom, reg_start, reg_end) in enumerate(x_regions):
        # Get bins for this region
        if i < len(x_boundaries) - 1:
            start_idx = x_boundaries[i]
            end_idx = x_boundaries[i + 1]
        else:
            start_idx = x_boundaries[i]
            end_idx = len(x_bins)
        
        region_bins = x_bins.iloc[start_idx:end_idx].reset_index(drop=True)
        
        # Generate ticks at regular intervals
        first_tick = int(np.ceil(reg_start / tick_interval) * tick_interval)
        tick_coords = np.arange(first_tick, reg_end, tick_interval)
        
        for tick_coord in tick_coords:
            # Find the bin that contains this coordinate
            matching_bins = region_bins[
                (region_bins['start'] <= tick_coord) & 
                (region_bins['end'] > tick_coord)
            ]
            
            if len(matching_bins) > 0:
                # Get position in the submatrix
                local_idx = matching_bins.index[0]
                matrix_idx = start_idx + local_idx
                
                x_tick_positions.append(matrix_idx)
                x_tick_labels.append(f"{tick_coord / 1e6:.2f}")
    
    # Generate coordinate ticks for y-axis
    y_tick_positions = []
    y_tick_labels = []
    
    for i, (chrom, reg_start, reg_end) in enumerate(y_regions):
        # Get bins for this region
        if i < len(y_boundaries) - 1:
            start_idx = y_boundaries[i]
            end_idx = y_boundaries[i + 1]
        else:
            start_idx = y_boundaries[i]
            end_idx = len(y_bins)
        
        region_bins = y_bins.iloc[start_idx:end_idx].reset_index(drop=True)
        
        # Generate ticks at regular intervals
        first_tick = int(np.ceil(reg_start / tick_interval) * tick_interval)
        tick_coords = np.arange(first_tick, reg_end, tick_interval)
        
        for tick_coord in tick_coords:
            # Find the bin that contains this coordinate
            matching_bins = region_bins[
                (region_bins['start'] <= tick_coord) & 
                (region_bins['end'] > tick_coord)
            ]
            
            if len(matching_bins) > 0:
                # Get position in the submatrix
                local_idx = matching_bins.index[0]
                matrix_idx = start_idx + local_idx
                
                y_tick_positions.append(matrix_idx)
                y_tick_labels.append(f"{tick_coord / 1e6:.2f}")
    
    # Set coordinate ticks on bottom and left
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels, fontsize=12)
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels, fontsize=12)
    
    # Add chromosome name labels on TOP (x-axis) and LEFT (y-axis)
    # X-axis chromosome labels (on top)
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    x_chr_positions = []
    x_chr_labels = []
    
    for i, (chrom, _, _) in enumerate(x_regions):
        if i < len(x_boundaries) - 1:
            # Middle of this chromosome region
            pos = (x_boundaries[i] + x_boundaries[i+1]) / 2
        else:
            # Last chromosome
            pos = (x_boundaries[i] + contact_matrix.shape[1]) / 2
        
        x_chr_positions.append(pos)
        x_chr_labels.append(chrom)
    
    ax_top.set_xticks(x_chr_positions)
    ax_top.set_xticklabels(x_chr_labels, fontsize=18, weight='bold')
    ax_top.tick_params(axis='x', length=0, pad=10)
    
    # Y-axis chromosome labels (on LEFT side, rotated 90 degrees)
    # Use the main ax object with secondary y-axis
    ax.text(-0.02, 0.5, '', transform=ax.transAxes)  # Spacer
    
    # Add text labels on the left for each chromosome
    for i, (chrom, _, _) in enumerate(y_regions):
        if i < len(y_boundaries) - 1:
            # Middle of this chromosome region
            pos = (y_boundaries[i] + y_boundaries[i+1]) / 2
        else:
            # Last chromosome
            pos = (y_boundaries[i] + contact_matrix.shape[0]) / 2
        
        # Convert to axis coordinates
        y_frac = pos / contact_matrix.shape[0]
        
        ax.text(-0.12, y_frac, chrom, 
                transform=ax.transAxes,
                fontsize=18, weight='bold',
                rotation=90, va='center', ha='center')
    
    # Bottom and left axis labels
    ax.set_xlabel('Position (Mb)', fontsize=20, weight='bold')
    ax.set_ylabel('Position (Mb)', fontsize=20, weight='bold')
    
    # Title
    ax.set_title('Hi-C Contact Map', fontsize=22, weight='bold', pad=40)
    
    plt.tight_layout()
    
    # Save PNG
    png_path = f"{output_path}.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    print(f"  Saved: {png_path}")
    
    # Save PDF
    pdf_path = f"{output_path}.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"  Saved: {pdf_path}")
    
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Plot Hi-C contact heatmap for custom genomic regions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single chromosome subregion (x and y axes)
  python plot_hic_subregion.py --bins bins.bed --matrix matrix.txt \\
      --x-coords "chr1:1000000-2000000" \\
      --y-coords "chr1:1000000-2000000" \\
      --output chr1_subregion
  
  # Multiple chromosomes on x-axis, single region on y-axis
  python plot_hic_subregion.py --bins bins.bed --matrix matrix.txt \\
      --x-coords "chr1:0-5000000,chr2:0-3000000" \\
      --y-coords "chrX:1000000-4000000" \\
      --output multi_chr_plot --vmax 500
  
  # Use percentile for color scale
  python plot_hic_subregion.py --bins bins.bed --matrix matrix.txt \\
      --x-coords "chr3:0-10000000" \\
      --y-coords "chr3:0-10000000" \\
      --output chr3_full --vmax p95
        """
    )
    
    parser.add_argument(
        '--bins',
        required=True,
        help='HiC-Pro bins file (4 columns: chrom, start, end, bin_id)'
    )
    
    parser.add_argument(
        '--matrix',
        required=True,
        help='HiC-Pro matrix file (sparse format: bin_i, bin_j, count)'
    )
    
    parser.add_argument(
        '--x-coords',
        required=True,
        help='Coordinates for x-axis (format: "chr:start-end,chr:start-end,...")'
    )
    
    parser.add_argument(
        '--y-coords',
        required=True,
        help='Coordinates for y-axis (format: "chr:start-end,chr:start-end,...")'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Output file prefix (without extension)'
    )
    
    parser.add_argument(
        '--vmax',
        default='20000',
        help='Maximum value for color scale. Options: "auto", "pNN" (percentile), or numeric value (default: 20000)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for PNG output (default: 300)'
    )
    
    parser.add_argument(
        '--tick-interval',
        type=int,
        default=100000,
        help='Interval for coordinate tick marks in bp (default: 100000 = 100kb)'
    )
    
    parser.add_argument(
        '--no-cpm',
        action='store_true',
        help='Skip CPM normalization (use raw counts)'
    )
    
    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_arguments()
    
    print("=" * 70)
    print("Hi-C SUBREGION HEATMAP")
    print("=" * 70)
    print(f"Bins file: {args.bins}")
    print(f"Matrix file: {args.matrix}")
    print(f"X-axis coords: {args.x_coords}")
    print(f"Y-axis coords: {args.y_coords}")
    print(f"CPM normalization: {not args.no_cpm}")
    print()
    
    # Parse coordinates
    print("PARSING COORDINATES")
    print("-" * 70)
    try:
        x_regions = parse_coordinates(args.x_coords)
        y_regions = parse_coordinates(args.y_coords)
        
        print(f"X-axis regions:")
        for chrom, start, end in x_regions:
            print(f"  {chrom}:{start:,}-{end:,}")
        
        print(f"Y-axis regions:")
        for chrom, start, end in y_regions:
            print(f"  {chrom}:{start:,}-{end:,}")
        
    except Exception as e:
        print(f"ERROR: Failed to parse coordinates: {e}")
        sys.exit(1)
    
    print()
    
    # Load data
    print("LOADING DATA")
    print("-" * 70)
    bins_df = load_bins_file(args.bins)
    matrix_df = load_matrix_file(args.matrix)
    print()
    
    # Select bins for regions
    print("SELECTING BINS")
    print("-" * 70)
    print("X-axis:")
    x_bins, x_boundaries = select_bins_for_regions(bins_df, x_regions)
    print(f"Y-axis:")
    y_bins, y_boundaries = select_bins_for_regions(bins_df, y_regions)
    print()
    
    # Build contact matrix
    print("BUILDING MATRIX")
    print("-" * 70)
    contact_matrix = build_submatrix(
        matrix_df, 
        x_bins, 
        y_bins, 
        cpm_normalize=not args.no_cpm
    )
    print()
    
    # Plot
    print("PLOTTING")
    print("-" * 70)
    plot_heatmap(
        contact_matrix,
        x_bins,
        y_bins,
        x_regions,
        y_regions,
        x_boundaries,
        y_boundaries,
        args.output,
        vmax=args.vmax,
        tick_interval=args.tick_interval,
        dpi=args.dpi
    )
    print()
    
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
