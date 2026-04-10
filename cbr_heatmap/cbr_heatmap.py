#!/usr/bin/env python3
"""
cbr_heatmap.py
==============
Generate Hi-C contact heatmaps organized by chromosome break regions (CBRs)
in Ascaris.  Builds a CBR-by-CBR contact matrix from HiC-Pro sparse matrices,
with regions sorted by type (terminal first, then internal) and within each
type by chromosome number.

Also produces PCA plots of trans-chromosomal interaction profiles (terminal
vs. internal) and trans interaction statistics comparing interaction strengths
between terminal-terminal, internal-internal, and terminal-internal CBR pairs.

Features:
  - CPM normalization
  - Optional within-chromosome masking (to isolate trans contacts)
  - Custom white→yellow→red→black colormap
  - PCA with 95% confidence ellipses
  - Saddle enrichment score (TT+II vs TI)
  - Multi-sample comparison mode

Dependencies:
  numpy, pandas, matplotlib, seaborn, scipy

Example usage:
  python cbr_heatmap.py \\
      --bed data/cbr_v50_200kb_split_internal.bed \\
      --bins data/AG_10kb_hicpro_bins.bed \\
      --matrix matrix/10kb/48hr_10kb_iced.matrix \\
      --output results/48hr \\
      --bin-size 10000 \\
      --vmaxes 'Vmax300:300' \\
      --colormap white_to_red \\
      --cpm-normalize \\
      --mask-chromosomes \\
      --trans-vmax 2000 \\
      --dpi 300

Input files:
  --bed: BED file defining CBR windows (chr start end name type).
      Type column = terminal or internal.
  --bins: HiC-Pro bins BED file (chr start end bin_id).
  --matrix: HiC-Pro ICE-normalized sparse matrix.
  These are intermediate HiC-Pro outputs; raw reads are on SRA (SRPXXXXXX).
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

plt.rcParams['svg.fonttype'] = 'none'  # editable text in SVG for Illustrator


# ============================================================================
# COLORMAP DEFINITIONS
# ============================================================================

def create_white_to_red_colormap():
    """
    Create custom colormap: white → yellow → orange → red → dark red → black.
    
    This is standard for Hi-C visualization where white represents no/low contact
    and darker colors represent stronger interactions.
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
    # Set bad (NaN) values to grey instead of white
    cmap.set_bad(color='#CCCCCC')  # Medium grey for masked regions
    return cmap


def get_colormap(colormap_name):
    """Get colormap by name and set NaN color to grey."""
    if colormap_name == 'white_to_red':
        return create_white_to_red_colormap()
    else:
        cmap = plt.get_cmap(colormap_name)
        # Make a copy and set bad values to grey
        cmap = cmap.copy()
        cmap.set_bad(color='#CCCCCC')
        return cmap


# ============================================================================
# FILE I/O FUNCTIONS
# ============================================================================

def load_bed_file(bed_path):
    """
    Load BED file and extract region information.
    
    Expected columns:
    0: chromosome
    1: start
    2: end
    3: region_name (e.g., CBR_001)
    4: region_type (terminal or internal)
    
    Returns sorted DataFrame with terminal regions first, then internal.
    """
    print(f"  Loading BED file: {bed_path}")
    
    try:
        df = pd.read_csv(
            bed_path, 
            sep='\t', 
            header=None,
            names=['chrom', 'start', 'end', 'name', 'type']
        )
    except Exception as e:
        print(f"ERROR: Failed to load BED file: {e}")
        sys.exit(1)
    
    # Filter out chrUn scaffolds
    n_before = len(df)
    df = df[~df['chrom'].str.contains('chrUn', na=False)]
    n_filtered = n_before - len(df)
    
    if n_filtered > 0:
        print(f"  Filtered {n_filtered} chrUn scaffold regions")
    
    # Sort by type (terminal first, then internal) and then by chromosome/position
    df['type_order'] = df['type'].map({'terminal': 0, 'internal': 1})
    df = df.sort_values(['type_order', 'chrom', 'start']).reset_index(drop=True)
    df = df.drop('type_order', axis=1)
    
    print(f"  Loaded {len(df)} regions:")
    type_counts = df['type'].value_counts()
    for region_type, count in type_counts.items():
        print(f"    - {region_type}: {count}")
    
    return df


def load_bins_file(bins_path):
    """
    Load HiC-Pro bins file.
    
    Expected columns:
    0: chromosome
    1: start
    2: end
    3: bin_id
    """
    print(f"  Loading bins file: {bins_path}")
    
    try:
        df = pd.read_csv(
            bins_path,
            sep='\t',
            header=None,
            names=['chrom', 'start', 'end', 'bin_id']
        )
    except Exception as e:
        print(f"ERROR: Failed to load bins file: {e}")
        sys.exit(1)
    
    print(f"  Loaded {len(df)} bins")
    return df


def load_matrix_file(matrix_path):
    """
    Load HiC-Pro matrix file in sparse format.
    
    Expected columns:
    0: bin_i (1-indexed)
    1: bin_j (1-indexed) 
    2: count
    
    Returns DataFrame with 0-indexed bins.
    """
    print(f"  Loading matrix file: {matrix_path}")
    
    try:
        df = pd.read_csv(
            matrix_path,
            sep='\t',
            header=None,
            names=['bin_i', 'bin_j', 'count']
        )
        
        # Convert to 0-indexed
        df['bin_i'] -= 1
        df['bin_j'] -= 1
        
    except Exception as e:
        print(f"ERROR: Failed to load matrix file: {e}")
        sys.exit(1)
    
    print(f"  Loaded {len(df):,} interactions")
    print(f"  Total contacts: {df['count'].sum():,.0f}")
    
    return df


# ============================================================================
# PLOTTING HELPER FUNCTIONS
# ============================================================================

def draw_significance_bar(ax, x1, x2, y, p_value, bar_height=0.02):
    """
    Draw a significance bar between two x positions with p-value annotation.
    
    Args:
        ax: Matplotlib axis object
        x1, x2: X positions for the two groups
        y: Y position for the bar (in data coordinates)
        p_value: P-value to display
        bar_height: Height of the bar endpoints relative to y-axis range
    """
    # Get y-axis range for scaling bar height
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    h = bar_height * y_range
    
    # Draw the bar
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2, c='black')
    
    # Format p-value text
    if p_value < 0.001:
        p_text = '***'
    elif p_value < 0.01:
        p_text = '**'
    elif p_value < 0.05:
        p_text = '*'
    else:
        p_text = 'ns'
    
    # Add p-value text
    x_center = (x1 + x2) / 2
    ax.text(x_center, y+h, p_text, ha='center', va='bottom', fontsize=14, weight='bold')


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def map_bins_to_regions(bins_df, regions_df):
    """
    Map genomic bins to BED regions based on overlap.
    
    Returns:
    - bin_to_region: dict mapping bin_id to region index
    - region_bins: list of lists, bins for each region
    """
    print("  Mapping bins to regions...")
    
    bin_to_region = {}
    region_bins = [[] for _ in range(len(regions_df))]
    
    for region_idx, region in regions_df.iterrows():
        # Find bins that overlap this region
        overlapping_bins = bins_df[
            (bins_df['chrom'] == region['chrom']) &
            (bins_df['start'] < region['end']) &
            (bins_df['end'] > region['start'])
        ]
        
        for _, bin_row in overlapping_bins.iterrows():
            bin_id = bin_row['bin_id']
            bin_to_region[bin_id] = region_idx
            region_bins[region_idx].append(bin_id)
    
    # Count bins per region
    n_mapped = sum(len(bins) for bins in region_bins)
    print(f"  Mapped {n_mapped} bins to {len(regions_df)} regions")
    
    # Check for regions with no bins
    empty_regions = sum(1 for bins in region_bins if len(bins) == 0)
    if empty_regions > 0:
        print(f"  WARNING: {empty_regions} regions have no bins")
    
    return bin_to_region, region_bins


def build_contact_matrix(matrix_df, bin_to_region, n_regions, 
                         mask_chromosomes=False,
                         regions_df=None):
    """
    Build region-by-region contact matrix from bin-level data.
    
    Args:
        matrix_df: Sparse matrix DataFrame
        bin_to_region: Mapping of bin IDs to region indices
        n_regions: Total number of regions
        mask_chromosomes: If True, set within-chromosome contacts to NaN
        regions_df: Region DataFrame (needed for chromosome masking)
    
    Returns:
        Symmetric contact matrix (numpy array)
    """
    print("  Building region contact matrix...")
    
    # Initialize matrix
    contact_matrix = np.zeros((n_regions, n_regions), dtype=np.float64)
    
    # Aggregate contacts
    n_used = 0
    for _, row in matrix_df.iterrows():
        bin_i = int(row['bin_i'])
        bin_j = int(row['bin_j'])
        count = row['count']
        
        # Check if both bins are in regions
        if bin_i in bin_to_region and bin_j in bin_to_region:
            region_i = bin_to_region[bin_i]
            region_j = bin_to_region[bin_j]
            
            # Add to matrix (symmetrically)
            contact_matrix[region_i, region_j] += count
            if region_i != region_j:
                contact_matrix[region_j, region_i] += count
            
            n_used += 1
    
    print(f"  Used {n_used:,} interactions ({n_used/len(matrix_df)*100:.1f}%)")
    
    # Apply chromosome masking if requested
    if mask_chromosomes and regions_df is not None:
        print("  Masking within-chromosome interactions...")
        for i in range(n_regions):
            for j in range(n_regions):
                if regions_df.iloc[i]['chrom'] == regions_df.iloc[j]['chrom']:
                    contact_matrix[i, j] = np.nan
    
    return contact_matrix


def normalize_cpm(contact_matrix):
    """
    Normalize contact matrix to counts per million (CPM).
    
    CPM = (count / total_counts) * 1,000,000
    """
    print("  Applying CPM normalization...")
    
    # Get total valid (non-NaN) contacts
    total_contacts = np.nansum(contact_matrix)
    
    if total_contacts == 0:
        print("  WARNING: No contacts to normalize")
        return contact_matrix
    
    # Normalize
    normalized = (contact_matrix / total_contacts) * 1e6
    
    print(f"  Total contacts: {total_contacts:,.0f}")
    print(f"  CPM range: {np.nanmin(normalized):.2e} - {np.nanmax(normalized):.2e}")
    
    return normalized


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_region_labels(regions_df, max_label_length=15):
    """
    Create informative labels for regions showing only chromosome names.
    
    Format: Just the region name (e.g., "chr01.2") without type indicators.
    """
    labels = []
    
    for _, region in regions_df.iterrows():
        name = region['name']
        
        # Shorten long names if needed
        if len(name) > max_label_length:
            name = name[:max_label_length-3] + "..."
        
        # Just use the name without any type indicator
        labels.append(name)
    
    return labels


def calculate_vmax(contact_matrix, vmax_spec):
    """
    Calculate vmax from specification.
    
    Args:
        contact_matrix: Contact frequency matrix
        vmax_spec: Either 'auto', 'pNN' (percentile), or numeric value
    
    Returns:
        Numeric vmax value
    """
    if vmax_spec == 'auto':
        # Use maximum value in data
        return np.nanmax(contact_matrix)
    elif vmax_spec.startswith('p'):
        # Percentile-based (e.g., 'p95', 'p99')
        try:
            percentile = float(vmax_spec[1:])
            return np.nanpercentile(contact_matrix, percentile)
        except:
            print(f"    WARNING: Invalid percentile spec '{vmax_spec}', using auto")
            return np.nanmax(contact_matrix)
    else:
        # Numeric value
        try:
            return float(vmax_spec)
        except:
            print(f"    WARNING: Invalid vmax spec '{vmax_spec}', using auto")
            return np.nanmax(contact_matrix)


def plot_heatmap(contact_matrix, regions_df, output_path, 
                vmax_spec='auto', vmax_value=None, colormap='white_to_red', dpi=300):
    """
    Generate Hi-C contact heatmap with specified visualization range.
    
    Args:
        contact_matrix: Region contact matrix
        regions_df: DataFrame with region information
        output_path: Path for output file (without extension)
        vmax_spec: Specification for vmax ('auto', 'pNN', or numeric)
        vmax_value: Pre-calculated vmax value (if None, will calculate from vmax_spec)
        colormap: Name of colormap to use
        dpi: Resolution for PNG output
    """
    print(f"    Generating plot (vmax={vmax_spec})...")
    
    # Calculate vmax if not provided
    if vmax_value is None:
        vmax_value = calculate_vmax(contact_matrix, vmax_spec)
    
    print(f"    Using vmax={vmax_value:.2e}")
    
    # Set up figure with appropriate size
    n_regions = len(regions_df)
    figsize = max(12, n_regions * 0.3)  # Scale with number of regions
    
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    
    # Create labels (chromosome names only, no T/I indicators)
    labels = create_region_labels(regions_df)
    
    # Get colormap with grey for NaN values
    cmap = get_colormap(colormap)
    
    # Define linear color normalization (standard for Hi-C)
    norm = Normalize(vmin=0, vmax=vmax_value)
    
    # Create heatmap
    im = ax.imshow(
        contact_matrix,
        cmap=cmap,
        norm=norm,
        aspect='auto',
        interpolation='nearest'
    )
    
    # Add colorbar with large label
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Contact Frequency (CPM)', fontsize=20, weight='bold')
    cbar.ax.tick_params(labelsize=16)
    
    # Set axis labels with region names
    ax.set_xticks(range(n_regions))
    ax.set_yticks(range(n_regions))
    ax.set_xticklabels(labels, rotation=90, ha='right', fontsize=14)
    ax.set_yticklabels(labels, fontsize=14)
    
    # Add grid lines between region types
    terminal_count = (regions_df['type'] == 'terminal').sum()
    if 0 < terminal_count < n_regions:
        ax.axhline(y=terminal_count - 0.5, color='blue', linewidth=3, alpha=0.7)
        ax.axvline(x=terminal_count - 0.5, color='blue', linewidth=3, alpha=0.7)
    
    # Add title with large font
    ax.set_title(
        f'Hi-C Contact Heatmap (vmax={vmax_value:.2e})',
        fontsize=24,
        weight='bold',
        pad=20
    )
    
    # Add axis labels - changed from "Region" to "Chromosome Breakage Region (CBR)"
    ax.set_xlabel('Chromosome Breakage Region (CBR)', fontsize=20, weight='bold')
    ax.set_ylabel('Chromosome Breakage Region (CBR)', fontsize=20, weight='bold')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save PNG
    png_path = f"{output_path}.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    print(f"    Saved PNG: {png_path}")
    
    # Save PDF (vector format)
    pdf_path = f"{output_path}.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"    Saved PDF: {pdf_path}")
    
    plt.close(fig)


def plot_trans_interaction_stats(contact_matrix, regions_df, output_path, vmax=2000, dpi=300):
    """
    Analyze and plot trans-chromosomal interaction strengths grouped by region type.
    
    Creates a dot plot showing:
    - Terminal-to-terminal interactions (dark red)
    - Internal-to-internal interactions (red)
    - Terminal-to-internal interactions (light red)
    
    Points above vmax are shown as empty circles, with outlier counts reported.
    Displays mean ± SEM and performs Kruskal-Wallis test.
    
    Args:
        contact_matrix: Region contact matrix
        regions_df: DataFrame with region information (must have 'type' and 'chrom' columns)
        output_path: Path for output file (without extension)
        vmax: Maximum value for y-axis; points above shown as empty circles (default: 3000)
        dpi: Resolution for PNG output
    """
    print(f"    Generating trans-interaction statistics plot (vmax={vmax})...")
    
    n_regions = len(regions_df)
    
    # Initialize lists to store interactions by type
    tt_interactions = []  # Terminal-to-terminal
    ii_interactions = []  # Internal-to-internal
    ti_interactions = []  # Terminal-to-internal
    
    # Extract trans-chromosomal interactions from upper triangle
    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            # Check if trans-chromosomal (different chromosomes)
            if regions_df.iloc[i]['chrom'] != regions_df.iloc[j]['chrom']:
                value = contact_matrix[i, j]
                
                # Skip NaN values
                if np.isnan(value):
                    continue
                
                # Determine interaction type
                type_i = regions_df.iloc[i]['type']
                type_j = regions_df.iloc[j]['type']
                
                if type_i == 'terminal' and type_j == 'terminal':
                    tt_interactions.append(value)
                elif type_i == 'internal' and type_j == 'internal':
                    ii_interactions.append(value)
                else:  # One terminal, one internal
                    ti_interactions.append(value)
    
    # Convert to arrays
    tt_interactions = np.array(tt_interactions)
    ii_interactions = np.array(ii_interactions)
    ti_interactions = np.array(ti_interactions)
    
    # Check if we have data
    if len(tt_interactions) == 0 and len(ii_interactions) == 0 and len(ti_interactions) == 0:
        print("    WARNING: No trans-chromosomal interactions found")
        return
    
    # Count outliers above vmax
    tt_outliers = np.sum(tt_interactions > vmax) if len(tt_interactions) > 0 else 0
    ii_outliers = np.sum(ii_interactions > vmax) if len(ii_interactions) > 0 else 0
    ti_outliers = np.sum(ti_interactions > vmax) if len(ti_interactions) > 0 else 0
    total_outliers = tt_outliers + ii_outliers + ti_outliers
    
    # Calculate statistics
    tt_mean = np.mean(tt_interactions) if len(tt_interactions) > 0 else np.nan
    tt_sem = stats.sem(tt_interactions) if len(tt_interactions) > 0 else np.nan
    ii_mean = np.mean(ii_interactions) if len(ii_interactions) > 0 else np.nan
    ii_sem = stats.sem(ii_interactions) if len(ii_interactions) > 0 else np.nan
    ti_mean = np.mean(ti_interactions) if len(ti_interactions) > 0 else np.nan
    ti_sem = stats.sem(ti_interactions) if len(ti_interactions) > 0 else np.nan
    
    # Perform Kruskal-Wallis test (non-parametric test for 3+ groups)
    groups_with_data = []
    group_names = []
    if len(tt_interactions) > 0:
        groups_with_data.append(tt_interactions)
        group_names.append('TT')
    if len(ii_interactions) > 0:
        groups_with_data.append(ii_interactions)
        group_names.append('II')
    if len(ti_interactions) > 0:
        groups_with_data.append(ti_interactions)
        group_names.append('TI')
    
    if len(groups_with_data) >= 2:
        h_stat, kw_p_value = stats.kruskal(*groups_with_data)
    else:
        h_stat, kw_p_value = np.nan, np.nan
    
    # Perform pairwise Mann-Whitney U tests with Bonferroni correction
    pairwise_results = {}
    if len(groups_with_data) == 3:
        # All three groups present - do 3 pairwise comparisons
        comparisons = [
            ('TT', 'II', tt_interactions, ii_interactions),
            ('TT', 'TI', tt_interactions, ti_interactions),
            ('II', 'TI', ii_interactions, ti_interactions)
        ]
        bonferroni_factor = 3
        
        for name1, name2, data1, data2 in comparisons:
            if len(data1) > 0 and len(data2) > 0:
                u_stat, p_val = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                p_val_corrected = min(p_val * bonferroni_factor, 1.0)  # Bonferroni correction
                pairwise_results[f'{name1}_vs_{name2}'] = p_val_corrected
    elif len(groups_with_data) == 2:
        # Only two groups - do 1 comparison (no correction needed)
        if len(groups_with_data[0]) > 0 and len(groups_with_data[1]) > 0:
            u_stat, p_val = stats.mannwhitneyu(groups_with_data[0], groups_with_data[1], alternative='two-sided')
            pairwise_results[f'{group_names[0]}_vs_{group_names[1]}'] = p_val
    
    # Print statistics
    print(f"    Trans-chromosomal interaction statistics:")
    print(f"      Terminal-Terminal: n={len(tt_interactions)}, mean={tt_mean:.2f}, SEM={tt_sem:.2f}, outliers={tt_outliers}")
    print(f"      Internal-Internal: n={len(ii_interactions)}, mean={ii_mean:.2f}, SEM={ii_sem:.2f}, outliers={ii_outliers}")
    print(f"      Terminal-Internal: n={len(ti_interactions)}, mean={ti_mean:.2f}, SEM={ti_sem:.2f}, outliers={ti_outliers}")
    if not np.isnan(kw_p_value):
        print(f"      Kruskal-Wallis H={h_stat:.2f}, p={kw_p_value:.2e}")
    for comparison, p_val in pairwise_results.items():
        print(f"      {comparison}: p={p_val:.2e}")
    
    # Define colors: dark red, red, light red
    colors = {
        'Terminal-Terminal': '#8B0000',  # Dark red
        'Internal-Internal': '#FF0000',  # Red
        'Terminal-Internal': '#FF6B6B'   # Light red
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Plot data for each group
    groups_data = [
        ('Terminal-Terminal', tt_interactions, colors['Terminal-Terminal']),
        ('Internal-Internal', ii_interactions, colors['Internal-Internal']),
        ('Terminal-Internal', ti_interactions, colors['Terminal-Internal'])
    ]
    
    x_positions = []
    means_plot = []
    sems_plot = []
    
    for idx, (label, data, color) in enumerate(groups_data):
        if len(data) == 0:
            continue
            
        x_pos = idx
        x_positions.append(x_pos)
        
        # Separate data into in-range and outliers
        in_range = data[data <= vmax]
        outliers = data[data > vmax]
        
        # Plot in-range points as filled circles
        if len(in_range) > 0:
            x_jitter = np.random.normal(x_pos, 0.08, size=len(in_range))
            ax.scatter(x_jitter, in_range, s=30, alpha=0.4, color=color, 
                      edgecolors='none', zorder=1)
        
        # Plot outliers as empty circles at vmax
        if len(outliers) > 0:
            x_jitter = np.random.normal(x_pos, 0.08, size=len(outliers))
            ax.scatter(x_jitter, [vmax] * len(outliers), s=40, alpha=0.6, 
                      facecolors='none', edgecolors=color, linewidths=1.5, zorder=1)
        
        # Calculate and store mean/SEM
        mean_val = np.mean(data)
        sem_val = stats.sem(data)
        means_plot.append(mean_val)
        sems_plot.append(sem_val)
    
    # Plot means as larger filled circles (capped at vmax for display)
    means_display = [min(m, vmax) for m in means_plot]
    ax.scatter(x_positions, means_display, s=100, color='black', 
              marker='o', zorder=10, edgecolors='white', linewidths=2,
              label='Mean')
    
    # Plot error bars (capped at vmax)
    for x_pos, mean_val, sem_val in zip(x_positions, means_plot, sems_plot):
        # Calculate error bar bounds
        lower = mean_val - sem_val
        upper = mean_val + sem_val
        
        # Cap at vmax for display
        if mean_val > vmax:
            # Mean is above vmax, don't show error bar
            continue
        elif upper > vmax:
            # Upper bound exceeds vmax, cap it
            upper = vmax
        
        # Draw error bar
        ax.plot([x_pos, x_pos], [lower, upper], 'k-', linewidth=3, zorder=9)
        ax.plot([x_pos - 0.1, x_pos + 0.1], [lower, lower], 'k-', linewidth=3, zorder=9)
        ax.plot([x_pos - 0.1, x_pos + 0.1], [upper, upper], 'k-', linewidth=3, zorder=9)
    
    # Format plot
    ax.set_ylabel('Contact Frequency (CPM)', fontsize=20, weight='bold')
    ax.set_xlabel('Interaction Type', fontsize=20, weight='bold')
    ax.tick_params(axis='both', labelsize=16)
    
    # Set x-axis labels with n and outliers
    group_labels_with_stats = []
    group_stats = [
        ('Terminal-Terminal', len(tt_interactions), tt_outliers),
        ('Internal-Internal', len(ii_interactions), ii_outliers),
        ('Terminal-Internal', len(ti_interactions), ti_outliers)
    ]
    for label, n, outliers in group_stats:
        if n > 0:  # Only add if this group has data
            group_labels_with_stats.append(f'{label}\n(n={n}, outliers={outliers})')
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(group_labels_with_stats, fontsize=12)
    
    # Set y-axis limits (add extra space for significance bars if present)
    if len(pairwise_results) > 0:
        y_max = vmax * 1.35  # Extra space for significance bars
    else:
        y_max = vmax * 1.05
    ax.set_ylim(0, y_max)
    
    # Add horizontal line at vmax
    ax.axhline(y=vmax, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    
    # Add significance bars with p-values
    if len(pairwise_results) > 0:
        # Find the actual maximum data point value (not capped at vmax)
        all_data_values = []
        for label, data, color in groups_data:
            if len(data) > 0:
                all_data_values.extend(data)
        
        max_actual_y = max(all_data_values) if all_data_values else vmax
        
        # Position bars above the highest data point
        bar_y_start = min(max_actual_y * 1.05, vmax * 0.85)  # Start at 105% of max data or 85% of vmax, whichever is lower
        bar_y_spacing = vmax * 0.08
        
        # Map comparison keys to x-position pairs
        comparison_map = {
            'TT_vs_II': (0, 1),
            'TT_vs_TI': (0, 2),
            'II_vs_TI': (1, 2)
        }
        
        # Sort comparisons by x-distance (smaller distance = lower bar)
        sorted_comparisons = sorted(pairwise_results.items(), 
                                   key=lambda x: abs(comparison_map.get(x[0], (0, 0))[1] - 
                                                   comparison_map.get(x[0], (0, 0))[0]))
        
        for i, (comp_key, p_val) in enumerate(sorted_comparisons):
            if comp_key in comparison_map:
                x1, x2 = comparison_map[comp_key]
                y_pos = bar_y_start + i * bar_y_spacing
                draw_significance_bar(ax, x1, x2, y_pos, p_val)
    
    # Add title with pairwise statistics
    title_lines = ['Trans-Chromosomal Interaction Strengths']
    
    # Format pairwise p-values for title
    if len(pairwise_results) > 0:
        p_texts = []
        comparison_labels = {
            'TT_vs_II': 'TT vs II',
            'TT_vs_TI': 'TT vs TI',
            'II_vs_TI': 'II vs TI'
        }
        for comp_key, p_val in pairwise_results.items():
            comp_label = comparison_labels.get(comp_key, comp_key)
            if p_val < 0.001:
                p_texts.append(f'{comp_label}: p<0.001')
            else:
                p_texts.append(f'{comp_label}: p={p_val:.3f}')
        title_lines.append(' | '.join(p_texts))
    
    # Add outlier count
    if total_outliers > 0:
        title_lines.append(f'({total_outliers} outliers > {vmax} CPM shown as empty circles)')
    
    ax.set_title('\n'.join(title_lines), fontsize=16, weight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                   markersize=12, markeredgecolor='white', markeredgewidth=2, label='Mean ± SEM'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=8, alpha=0.4, label='Data points'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
                   markeredgecolor='gray', markersize=8, markeredgewidth=1.5, 
                   label=f'Outliers (n={total_outliers})')
    ]
    ax.legend(handles=legend_elements, fontsize=12, loc='upper right')
    
    # Add grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save PNG
    png_path = f"{output_path}_trans_stats.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    print(f"    Saved PNG: {png_path}")
    
    # Save PDF
    pdf_path = f"{output_path}_trans_stats.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"    Saved PDF: {pdf_path}")
    
    # Save statistics to text file
    stats_path = f"{output_path}_trans_stats.txt"
    with open(stats_path, 'w') as f:
        f.write("Trans-Chromosomal Interaction Statistics\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Plot Settings:\n")
        f.write(f"  vmax = {vmax} CPM\n")
        f.write(f"  Total outliers above vmax = {total_outliers}\n\n")
        f.write(f"Terminal-Terminal:\n")
        f.write(f"  n = {len(tt_interactions)}\n")
        f.write(f"  Mean = {tt_mean:.4f} CPM\n")
        f.write(f"  SEM = {tt_sem:.4f}\n")
        f.write(f"  Outliers > {vmax} = {tt_outliers}\n\n")
        f.write(f"Internal-Internal:\n")
        f.write(f"  n = {len(ii_interactions)}\n")
        f.write(f"  Mean = {ii_mean:.4f} CPM\n")
        f.write(f"  SEM = {ii_sem:.4f}\n")
        f.write(f"  Outliers > {vmax} = {ii_outliers}\n\n")
        f.write(f"Terminal-Internal:\n")
        f.write(f"  n = {len(ti_interactions)}\n")
        f.write(f"  Mean = {ti_mean:.4f} CPM\n")
        f.write(f"  SEM = {ti_sem:.4f}\n")
        f.write(f"  Outliers > {vmax} = {ti_outliers}\n\n")
        if not np.isnan(p_value):
            f.write(f"Kruskal-Wallis Test:\n")
            f.write(f"  H-statistic = {h_stat:.4f}\n")
            f.write(f"  p-value = {p_value:.6e}\n")
            if p_value < 0.05:
                f.write(f"  Result: Significant difference between groups (p < 0.05)\n")
            else:
                f.write(f"  Result: No significant difference between groups (p ≥ 0.05)\n")
    
    print(f"    Saved statistics: {stats_path}")
    
    plt.close(fig)


def calculate_saddle_score(contact_matrix, regions_df):
    """
    Calculate saddle-like enrichment score for self vs cross interactions.
    
    NOTE: This is not a true compartment analysis (no eigenvector calculation).
    Instead, it quantifies whether same-type regions (terminal-terminal, internal-internal)
    interact more strongly than different-type regions (terminal-internal).
    
    Score = (mean_TT + mean_II) / (2 * mean_TI)
    
    IMPORTANT: This does NOT account for different numbers of regions in each category.
    With unequal group sizes, the interpretation should be cautious.
    
    Args:
        contact_matrix: Region contact matrix
        regions_df: DataFrame with region information (must have 'type' and 'chrom' columns)
    
    Returns:
        Dictionary with score and component values
    """
    n_regions = len(regions_df)
    
    # Count terminal and internal regions
    n_terminal = sum(regions_df['type'] == 'terminal')
    n_internal = sum(regions_df['type'] == 'internal')
    
    # Initialize lists for each interaction type
    tt_values = []  # Terminal-terminal
    ii_values = []  # Internal-internal
    ti_values = []  # Terminal-internal
    
    # Extract trans-chromosomal interactions from upper triangle
    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            # Check if trans-chromosomal (different chromosomes)
            if regions_df.iloc[i]['chrom'] != regions_df.iloc[j]['chrom']:
                value = contact_matrix[i, j]
                
                # Skip NaN values
                if np.isnan(value):
                    continue
                
                # Determine interaction type
                type_i = regions_df.iloc[i]['type']
                type_j = regions_df.iloc[j]['type']
                
                if type_i == 'terminal' and type_j == 'terminal':
                    tt_values.append(value)
                elif type_i == 'internal' and type_j == 'internal':
                    ii_values.append(value)
                else:
                    ti_values.append(value)
    
    # Check if we have sufficient data
    if len(tt_values) == 0 or len(ii_values) == 0 or len(ti_values) == 0:
        return None
    
    # Calculate means
    mean_tt = np.mean(tt_values)
    mean_ii = np.mean(ii_values)
    mean_ti = np.mean(ti_values)
    
    # Calculate enrichment score
    self_interaction = mean_tt + mean_ii
    cross_interaction = 2 * mean_ti
    enrichment_score = self_interaction / cross_interaction
    
    return {
        'score': enrichment_score,
        'mean_TT': mean_tt,
        'mean_II': mean_ii,
        'mean_TI': mean_ti,
        'n_TT': len(tt_values),
        'n_II': len(ii_values),
        'n_TI': len(ti_values),
        'n_terminal_regions': n_terminal,
        'n_internal_regions': n_internal
    }


def save_saddle_score(result, output_path, sample_name):
    """
    Save saddle enrichment score to a file for later aggregation.
    
    Args:
        result: Dictionary from calculate_saddle_score()
        output_path: Base output path
        sample_name: Name of the sample
    """
    score_file = f"{output_path}_saddle_score.txt"
    
    with open(score_file, 'w') as f:
        f.write(f"{sample_name}\t{result['score']:.4f}\n")


def plot_saddle_comparison(output_base_dir, sample_names, dpi=300):
    """
    Create comparison plot of saddle enrichment scores across samples.
    
    Reads individual saddle score files and creates a bar plot.
    
    Args:
        output_base_dir: Directory containing individual sample outputs
        sample_names: List of sample names (e.g., ['48hr', '60hr', '5day'])
        dpi: Resolution for PNG output
    """
    from pathlib import Path
    
    # Collect scores from files
    scores = []
    labels = []
    
    base_path = Path(output_base_dir)
    
    for sample in sample_names:
        # Try to find the score file for this sample
        score_files = list(base_path.glob(f"*{sample}*saddle_score.txt"))
        
        if score_files:
            score_file = score_files[0]
            try:
                with open(score_file, 'r') as f:
                    line = f.readline().strip()
                    name, score = line.split('\t')
                    scores.append(float(score))
                    labels.append(sample)
            except Exception as e:
                print(f"    WARNING: Could not read {score_file}: {e}")
                continue
    
    if len(scores) == 0:
        print("    WARNING: No saddle scores found for comparison plot")
        return
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create bars with gradient colors
    colors = ['#8B0000', '#CC0000', '#FF4444'][:len(scores)]
    
    bars = ax.bar(range(len(scores)), scores, color=colors, 
                  edgecolor='black', linewidth=2, alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}',
                ha='center', va='bottom', fontsize=16, weight='bold')
    
    # Format plot
    ax.set_ylabel('Self-Interaction Enrichment Score', fontsize=20, weight='bold')
    ax.set_xlabel('Developmental Stage', fontsize=20, weight='bold')
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(labels, fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    
    ax.set_title('Self vs Cross-Type Interaction Enrichment\n(Terminal-Terminal + Internal-Internal) / (2 × Terminal-Internal)',
                 fontsize=16, weight='bold', pad=20)
    
    # Add reference line at 1.0
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.7, 
               label='No enrichment (score = 1)')
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=12, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_path = base_path / "saddle_enrichment_comparison"
    
    png_path = f"{output_path}.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    print(f"    Saved comparison plot: {png_path}")
    
    pdf_path = f"{output_path}.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"    Saved PDF: {pdf_path}")
    
    plt.close(fig)


def plot_pca_analysis(contact_matrix, regions_df, output_path, dpi=300):
    """
    Perform PCA on contact patterns and visualize with clusters.
    
    Each region is represented by its trans-chromosomal interaction profile.
    Uses numpy for PCA computation (no sklearn required).
    
    Args:
        contact_matrix: Region contact matrix
        regions_df: DataFrame with region information
        output_path: Path for output file (without extension)
        dpi: Resolution for PNG output
    """
    from matplotlib.patches import Ellipse
    
    print(f"    Generating PCA analysis...")
    
    n_regions = len(regions_df)
    
    # Extract trans-chromosomal interaction profiles for each region
    trans_profiles = []
    valid_indices = []
    
    for i in range(n_regions):
        profile = []
        for j in range(n_regions):
            if i != j and regions_df.iloc[i]['chrom'] != regions_df.iloc[j]['chrom']:
                value = contact_matrix[i, j]
                if not np.isnan(value):
                    profile.append(value)
        
        # Only include regions with sufficient trans data
        if len(profile) >= 10:  # Require at least 10 trans interactions
            # Pad or truncate to common length
            trans_profiles.append(profile)
            valid_indices.append(i)
    
    if len(trans_profiles) < 10:
        print("    WARNING: Insufficient data for PCA analysis")
        return
    
    # Find common length (minimum) and pad/truncate
    min_length = min(len(p) for p in trans_profiles)
    trans_profiles = np.array([p[:min_length] for p in trans_profiles])
    
    # Standardize features (mean=0, std=1) using numpy
    mean = trans_profiles.mean(axis=0)
    std = trans_profiles.std(axis=0)
    std[std == 0] = 1  # Avoid division by zero
    trans_profiles_scaled = (trans_profiles - mean) / std
    
    # Perform PCA using numpy
    # Center the data
    data_centered = trans_profiles_scaled - trans_profiles_scaled.mean(axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(data_centered.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Project data onto first 2 principal components
    pc_coords = data_centered.dot(eigenvectors[:, :2])
    
    # Calculate explained variance ratio
    explained_variance_ratio = eigenvalues / eigenvalues.sum()
    
    # Get region types for valid indices
    region_types = [regions_df.iloc[i]['type'] for i in valid_indices]
    
    # Separate by type
    terminal_mask = np.array([t == 'terminal' for t in region_types])
    internal_mask = ~terminal_mask
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot points
    terminal_color = '#8B0000'  # Dark red
    internal_color = '#4169E1'  # Royal blue
    
    ax.scatter(pc_coords[terminal_mask, 0], pc_coords[terminal_mask, 1],
               c=terminal_color, s=100, alpha=0.6, edgecolors='black',
               linewidths=1.5, label='Terminal regions')
    
    ax.scatter(pc_coords[internal_mask, 0], pc_coords[internal_mask, 1],
               c=internal_color, s=100, alpha=0.6, edgecolors='black',
               linewidths=1.5, label='Internal regions')
    
    # Draw confidence ellipses around clusters
    def draw_ellipse(ax, points, color, label):
        """Draw 95% confidence ellipse around cluster"""
        if len(points) < 3:
            return
        
        mean = points.mean(axis=0)
        cov = np.cov(points.T)
        
        # Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        # Calculate angle
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        
        # 95% confidence interval (chi-square with 2 DOF = 5.991)
        width, height = 2 * np.sqrt(5.991 * eigenvalues)
        
        ellipse = Ellipse(mean, width, height, angle=angle,
                         facecolor='none', edgecolor=color,
                         linewidth=3, linestyle='--', alpha=0.8)
        ax.add_patch(ellipse)
    
    # Draw ellipses
    if terminal_mask.sum() >= 3:
        draw_ellipse(ax, pc_coords[terminal_mask], terminal_color, 'Terminal')
    
    if internal_mask.sum() >= 3:
        draw_ellipse(ax, pc_coords[internal_mask], internal_color, 'Internal')
    
    # Format plot
    ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}% variance)', 
                  fontsize=20, weight='bold')
    ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}% variance)',
                  fontsize=20, weight='bold')
    ax.set_title('PCA of Trans-Chromosomal Interaction Profiles\n(95% Confidence Ellipses)',
                 fontsize=18, weight='bold', pad=20)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(fontsize=14, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save PNG
    png_path = f"{output_path}_pca.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    print(f"    Saved PNG: {png_path}")
    
    # Save PDF
    pdf_path = f"{output_path}_pca.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"    Saved PDF: {pdf_path}")
    
    # Save statistics
    stats_path = f"{output_path}_pca_stats.txt"
    with open(stats_path, 'w') as f:
        f.write("PCA Statistics\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Regions analyzed: {len(valid_indices)}\n")
        f.write(f"  Terminal: {terminal_mask.sum()}\n")
        f.write(f"  Internal: {internal_mask.sum()}\n\n")
        f.write(f"Variance explained:\n")
        f.write(f"  PC1: {explained_variance_ratio[0]*100:.2f}%\n")
        f.write(f"  PC2: {explained_variance_ratio[1]*100:.2f}%\n")
        f.write(f"  Total: {(explained_variance_ratio[0] + explained_variance_ratio[1])*100:.2f}%\n")
    
    print(f"    Saved statistics: {stats_path}")
    
    plt.close(fig)




def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate Hi-C contact heatmaps with region-based organization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--bed',
        required=True,
        help='BED file with regions (columns: chrom, start, end, name, type)'
    )
    
    parser.add_argument(
        '--bins',
        required=True,
        help='HiC-Pro bins file'
    )
    
    parser.add_argument(
        '--matrix',
        required=True,
        help='HiC-Pro matrix file (sparse format)'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Output file prefix (without extension)'
    )
    
    parser.add_argument(
        '--bin-size',
        type=int,
        default=20000,
        help='Bin size in bp (used for output directory naming)'
    )
    
    parser.add_argument(
        '--cpm-normalize',
        action='store_true',
        help='Apply counts per million (CPM) normalization'
    )
    
    parser.add_argument(
        '--mask-chromosomes',
        action='store_true',
        help='Mask within-chromosome interactions'
    )
    
    parser.add_argument(
        '--vmaxes',
        default='auto:auto,p95:p95,p99:p99',
        help='Visualization ranges as name:vmax pairs (comma-separated). '
             'Options: '
             '(1) "auto" for data maximum (e.g., "max:auto"), '
             '(2) "pNN" for Nth percentile (e.g., "p95:p95" or "p99:p99"), '
             '(3) numeric value for fixed vmax across all samples (e.g., "fixed:0.01" or "vmax330:330"). '
             'Multiple ranges can be comma-separated (e.g., "auto:auto,p95:p95,fixed:330")'
    )
    
    parser.add_argument(
        '--colormap',
        default='white_to_red',
        choices=['white_to_red', 'Reds', 'YlOrRd', 'hot_r'],
        help='Colormap for heatmap'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for PNG output'
    )
    
    parser.add_argument(
        '--trans-vmax',
        type=float,
        default=2000.0,
        help='Maximum value for trans-chromosomal statistics plot y-axis. '
             'Points above this value are shown as empty circles (default: 2000 CPM)'
    )
    
    parser.add_argument(
        '--comparison-mode',
        action='store_true',
        help='Generate comparison plot across all samples (reads *_trans_stats.txt files)'
    )
    
    parser.add_argument(
        '--comparison-dir',
        default=None,
        help='Directory containing trans_stats.txt files for comparison mode'
    )
    
    return parser.parse_args()


def create_output_directory(base_output, bin_size, mask_chromosomes):
    """
    Create output directory with informative name including bin size and masking options.
    
    Returns the output prefix with directory path.
    """
    # Convert bin size to readable format (e.g., 10000 -> 10kb)
    if bin_size >= 1000:
        bin_label = f"{bin_size//1000}kb"
    else:
        bin_label = f"{bin_size}bp"
    
    # Build masking suffix
    if mask_chromosomes:
        mask_suffix = "_mask_chroms"
    else:
        mask_suffix = "_no_mask"
    
    # Create directory name
    dir_name = f"hic_plots_{bin_label}{mask_suffix}"
    
    # Get base directory and filename
    base_path = Path(base_output)
    base_dir = base_path.parent
    base_name = base_path.name
    
    # Create full output directory
    output_dir = base_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Return full path with directory and base filename
    output_prefix = output_dir / base_name
    
    print(f"  Output directory: {output_dir}")
    
    return str(output_prefix)


def parse_trans_stats_file(filepath):
    """
    Parse trans_stats.txt file to extract mean CPM values.
    
    Returns:
        Dictionary with 'TT', 'II', 'TI' keys mapping to mean CPM values
    """
    stats = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the mean values
    for i, line in enumerate(lines):
        if 'Terminal-Terminal:' in line:
            for j in range(i+1, min(i+5, len(lines))):
                if 'Mean =' in lines[j]:
                    mean_str = lines[j].split('Mean =')[1].split('CPM')[0].strip()
                    stats['TT'] = float(mean_str)
                    break
        
        elif 'Internal-Internal:' in line:
            for j in range(i+1, min(i+5, len(lines))):
                if 'Mean =' in lines[j]:
                    mean_str = lines[j].split('Mean =')[1].split('CPM')[0].strip()
                    stats['II'] = float(mean_str)
                    break
        
        elif 'Terminal-Internal:' in line:
            for j in range(i+1, min(i+5, len(lines))):
                if 'Mean =' in lines[j]:
                    mean_str = lines[j].split('Mean =')[1].split('CPM')[0].strip()
                    stats['TI'] = float(mean_str)
                    break
    
    return stats


def plot_trans_comparison(input_dir, dpi=300):
    """
    Create comparison plot of mean CPM across all samples.
    
    Args:
        input_dir: Directory containing trans_stats.txt files
        dpi: Resolution for PNG output
    """
    from pathlib import Path
    
    print("  Collecting statistics from trans_stats.txt files...")
    
    stats_files = list(Path(input_dir).glob('*_trans_stats.txt'))
    
    if not stats_files:
        print(f"  ERROR: No trans_stats.txt files found in {input_dir}")
        return
    
    all_stats = {}
    
    for stats_file in sorted(stats_files):
        # Extract sample name from filename (e.g., "01_48hr_trans_stats.txt" -> "48hr")
        filename = stats_file.stem
        parts = filename.split('_')
        sample_name = None
        for part in parts:
            if 'hr' in part or 'day' in part:
                sample_name = part
                break
        
        if not sample_name:
            sample_name = '_'.join(parts[1:-2])
        
        try:
            stats = parse_trans_stats_file(stats_file)
            if 'TT' in stats and 'II' in stats and 'TI' in stats:
                all_stats[sample_name] = stats
                print(f"    {sample_name}: TT={stats['TT']:.1f}, II={stats['II']:.1f}, TI={stats['TI']:.1f} CPM")
        except Exception as e:
            print(f"    WARNING: Could not parse {stats_file}: {e}")
            continue
    
    if not all_stats:
        print("  ERROR: No valid statistics found")
        return
    
    # Sort by sample name
    def sort_key(item):
        name = item[0]
        if 'hr' in name:
            return (0, int(name.replace('hr', '')))
        elif 'day' in name:
            return (1, int(name.replace('day', '')))
        else:
            return (2, name)
    
    sorted_items = sorted(all_stats.items(), key=sort_key)
    sample_names = [item[0] for item in sorted_items]
    
    # Extract values
    tt_values = [item[1]['TT'] for item in sorted_items]
    ii_values = [item[1]['II'] for item in sorted_items]
    ti_values = [item[1]['TI'] for item in sorted_items]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 9))
    
    x = np.arange(len(sample_names))
    width = 0.25
    
    # Colors matching trans_stats plot
    color_tt = '#8B0000'
    color_ii = '#FF0000'
    color_ti = '#FF6B6B'
    
    # Create bars
    bars1 = ax.bar(x - width, tt_values, width, label='Terminal-Terminal',
                   color=color_tt, edgecolor='black', linewidth=1.5, alpha=0.9)
    bars2 = ax.bar(x, ii_values, width, label='Internal-Internal',
                   color=color_ii, edgecolor='black', linewidth=1.5, alpha=0.9)
    bars3 = ax.bar(x + width, ti_values, width, label='Terminal-Internal',
                   color=color_ti, edgecolor='black', linewidth=1.5, alpha=0.9)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=11, weight='bold')
    
    # Format
    ax.set_ylabel('Mean Contact Frequency (CPM)', fontsize=20, weight='bold')
    ax.set_xlabel('Developmental Stage', fontsize=20, weight='bold')
    ax.set_title('Trans-Chromosomal Interaction Strengths Across Development',
                 fontsize=18, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(sample_names, fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=14, loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(input_dir) / "trans_interaction_comparison"
    
    png_path = f"{output_path}.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    print(f"  Saved PNG: {png_path}")
    
    pdf_path = f"{output_path}.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"  Saved PDF: {pdf_path}")
    
    plt.close(fig)


def main():
    """Main analysis pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Handle comparison mode
    if args.comparison_mode:
        print("=" * 70)
        print("TRANS-CHROMOSOMAL INTERACTION COMPARISON")
        print("=" * 70)
        comparison_dir = args.comparison_dir if args.comparison_dir else args.output
        print(f"Input directory: {comparison_dir}")
        print()
        
        try:
            plot_trans_comparison(comparison_dir, dpi=args.dpi)
        except Exception as e:
            print(f"ERROR: Failed to generate comparison plot: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        print()
        print("=" * 70)
        print("COMPLETE")
        print("=" * 70)
        return
    
    print("=" * 70)
    print("Hi-C HEATMAP GENERATION")
    print("=" * 70)
    print(f"Input matrix: {args.matrix}")
    print(f"Bin size: {args.bin_size} bp")
    print(f"Mask chromosomes: {args.mask_chromosomes}")
    print(f"CPM normalization: {args.cpm_normalize}")
    print()
    
    # Create output directory with bin size and masking info
    print("SETUP")
    print("-" * 70)
    output_prefix = create_output_directory(
        args.output,
        args.bin_size,
        args.mask_chromosomes
    )
    print()
    
    # Load data
    print("LOADING DATA")
    print("-" * 70)
    regions_df = load_bed_file(args.bed)
    bins_df = load_bins_file(args.bins)
    matrix_df = load_matrix_file(args.matrix)
    print()
    
    # Map bins to regions
    print("PROCESSING")
    print("-" * 70)
    bin_to_region, region_bins = map_bins_to_regions(bins_df, regions_df)
    
    # Build contact matrix
    contact_matrix = build_contact_matrix(
        matrix_df,
        bin_to_region,
        len(regions_df),
        mask_chromosomes=args.mask_chromosomes,
        regions_df=regions_df
    )
    
    # Apply CPM normalization if requested
    if args.cpm_normalize:
        contact_matrix = normalize_cpm(contact_matrix)
    
    print()
    
    # Parse vmax specifications
    print("GENERATING PLOTS")
    print("-" * 70)
    vmax_specs = args.vmaxes.split(',')
    vmaxes = {}
    for spec in vmax_specs:
        name, vmax_str = spec.split(':')
        vmaxes[name.strip()] = vmax_str.strip()
    
    print(f"  Creating {len(vmaxes)} visualization ranges:")
    for name, vmax_str in vmaxes.items():
        if vmax_str == 'auto':
            print(f"    - {name}: auto-scaled to data maximum")
        elif vmax_str.startswith('p'):
            percentile = vmax_str[1:]
            print(f"    - {name}: {percentile}th percentile")
        else:
            print(f"    - {name}: vmax={float(vmax_str):.2e}")
    print()
    
    # Generate plots for each vmax specification
    for vmax_name, vmax_spec in vmaxes.items():
        print(f"  [{vmax_name}]")
        output_path = f"{output_prefix}_{vmax_name}"
        
        try:
            # Generate regular continuous heatmap
            plot_heatmap(
                contact_matrix,
                regions_df,
                output_path,
                vmax_spec=vmax_spec,
                colormap=args.colormap,
                dpi=args.dpi
            )
                
        except Exception as e:
            print(f"    ERROR: Failed to generate plot: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print()
    
    # Generate trans-chromosomal interaction statistics plot
    # Only if we have masked chromosomes (showing trans interactions)
    if args.mask_chromosomes:
        print("TRANS-CHROMOSOMAL STATISTICS")
        print("-" * 70)
        try:
            plot_trans_interaction_stats(
                contact_matrix,
                regions_df,
                output_prefix,
                vmax=args.trans_vmax,
                dpi=args.dpi
            )
        except Exception as e:
            print(f"  ERROR: Failed to generate trans-stats plot: {e}")
            import traceback
            traceback.print_exc()
        print()
        
        # Calculate and save saddle enrichment score
        print("SELF-INTERACTION ENRICHMENT ANALYSIS")
        print("-" * 70)
        try:
            # Extract sample name from output prefix
            from pathlib import Path
            sample_name = Path(output_prefix).name.split('_', 1)[-1] if '_' in Path(output_prefix).name else Path(output_prefix).name
            
            result = calculate_saddle_score(contact_matrix, regions_df)
            if result is not None:
                print(f"    Enrichment score: {result['score']:.3f}")
                print(f"    Terminal-Terminal: n={result['n_TT']}, mean={result['mean_TT']:.2f} CPM")
                print(f"    Internal-Internal: n={result['n_II']}, mean={result['mean_II']:.2f} CPM")
                print(f"    Terminal-Internal: n={result['n_TI']}, mean={result['mean_TI']:.2f} CPM")
                print(f"    NOTE: {result['n_terminal_regions']} terminal regions, {result['n_internal_regions']} internal regions")
                print(f"          Unequal group sizes affect interpretation")
                
                save_saddle_score(result, output_prefix, sample_name)
                print(f"    Saved score for later comparison")
            else:
                print(f"    WARNING: Insufficient data for enrichment calculation")
        except Exception as e:
            print(f"  ERROR: Failed enrichment analysis: {e}")
            import traceback
            traceback.print_exc()
        print()
        
        # Generate PCA analysis
        print("PCA ANALYSIS")
        print("-" * 70)
        try:
            plot_pca_analysis(
                contact_matrix,
                regions_df,
                output_prefix,
                dpi=args.dpi
            )
        except Exception as e:
            print(f"  ERROR: Failed to generate PCA plot: {e}")
            import traceback
            traceback.print_exc()
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
