#!/usr/bin/env python
"""
plot_hic_triangle_heatmaps.py
==================
Generate stacked triangular Hi-C heatmaps across developmental timepoints for
Parascaris univalens.  Visualizes chromatin interaction patterns in a specified
genomic region, with each timepoint rendered as an upward-pointing triangle and
stacked vertically for easy comparison.

Supports both raw contact frequency and observed/expected (O/E) visualization,
with chromosome-aware expected value calculation for post-PDE timepoints where
the germline chromosome has fragmented into somatic chromosomes.

Outputs (per plot type):
  - High-resolution PNG (DPI scaled to bin count)
  - TIFF with LZW compression
  - SVG with editable text for Adobe Illustrator

Dependencies:
  numpy, matplotlib, scipy

Example usage:
  # log2(O/E) and raw heatmaps for somatic chromosomes chrX7–chrX9 at 40kb:
  python plot_hic_stages.py \\
      --region chrX7-chrX9 \\
      --mapping data/pu_v2_germ_to_soma_mapping.bed \\
      --resolution 40000 \\
      --bin-bed data/40000/pu_v2_prepde_abs_40kb \\
      --soma-bin-bed data/40000/pu_v2_postpde_abs_40kb \\
      --timepoints '10hr,17hr,24hr,48hr,72hr' \\
      --plot-type both \\
      --vmax 15 \\
      --vmin 0.0 \\
      --vmax-obsexp 2.5 \\
      --log2-obsexp \\
      --no-outline \\
      --output-dir results/

  # Raw heatmap only, direct coordinate range:
  python plot_hic_stages.py \\
      --region 10000000-15000000 \\
      --mapping data/pu_v2_germ_to_soma_mapping.bed \\
      --resolution 20000 \\
      --bin-bed data/20000/pu_v2_prepde_abs_20kb \\
      --timepoints '10hr,17hr,24hr,48hr,72hr' \\
      --plot-type raw \\
      --vmax 100 \\
      --output-dir results/

Region specification:
  --region accepts two formats:
    1. Somatic chromosome range: "chrX7-chrX9" (converted to germline coords
       via --mapping file)
    2. Direct germline coordinates: "10000000-15000000"

Input files:
  --mapping: Germline-to-somatic chromosome mapping (tab-delimited:
      germ_chr  start  end  soma_chr).  Required for chromosome-name regions
      and for post-PDE expected value calculation.
  --bin-bed: HiC-Pro abs.bed for the germline genome (pre-PDE bin mapping).
  --soma-bin-bed: HiC-Pro abs.bed for the somatic genome (post-PDE bin
      mapping).  Required if post-PDE timepoints are included and a somatic
      chromosome range is specified.
  Hi-C matrices: ICE-normalized sparse matrices from HiC-Pro, expected at
      matrix_files_{resolution}/prepde/ and matrix_files_{resolution}/postpde/.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
import argparse
from scipy.sparse import coo_matrix
import gc

def read_chromosome_mapping(mapping_file):
    """Read germline to somatic chromosome mapping"""
    mappings = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                soma_chr = parts[3]
                mappings[soma_chr] = {
                    'germ_chrom': chrom,
                    'germ_start': start,
                    'germ_end': end
                }
    return mappings

def get_somatic_region(soma_chr_start, soma_chr_end, mappings):
    """
    Get the somatic chromosome range for post-PDE samples
    Returns list of (chrom, start, end) tuples
    Note: For somatic genome, we typically want full chromosomes
    The end coordinate is approximate based on germline mapping
    """
    regions = []
    in_range = False
    
    # Get sorted chromosome list
    chr_list = sorted([k for k in mappings.keys() if k.startswith('chr')],
                     key=lambda x: (x.replace('chr0', 'chr').replace('chrX', 'chrZ'), x))
    
    for chr_name in chr_list:
        if chr_name == soma_chr_start:
            in_range = True
        
        if in_range:
            # For somatic genome, assume full chromosome
            # Use a large end position (will be constrained by actual bin mapping)
            regions.append((chr_name, 0, 1000000000))  # 1Gb max
        
        if chr_name == soma_chr_end:
            break
    
    return regions

def parse_region_arg(region_str, mapping_file):
    """
    Parse region argument like 'chr03-chr08' or '10000000-15000000'
    Returns (start_coord, end_coord)
    """
    if '-' not in region_str:
        raise ValueError("Region must be in format 'chr03-chr08' or '10000000-15000000'")
    
    parts = region_str.split('-')
    if len(parts) != 2:
        raise ValueError("Region must have exactly one '-' separator")
    
    # Check if it's chromosome names or coordinates
    if parts[0].startswith('chr') and parts[1].startswith('chr'):
        # Chromosome range
        mappings = read_chromosome_mapping(mapping_file)
        if parts[0] not in mappings or parts[1] not in mappings:
            available = sorted([k for k in mappings.keys() if k.startswith('chr0') or k.startswith('chrX')])
            raise ValueError(f"Invalid chromosome. Available: {', '.join(available)}")
        
        start_coord = mappings[parts[0]]['germ_start']
        end_coord = mappings[parts[1]]['germ_end']
        return start_coord, end_coord
    else:
        # Direct coordinates
        return int(parts[0]), int(parts[1])

def read_bin_mapping(bed_file, resolution=20000):
    """Read bin mapping from bed file"""
    bin_to_coords = {}
    coord_to_bin = {}
    chrom_bins = {}  # Track bins for each chromosome
    
    with open(bed_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                bin_num = int(parts[3])
                
                bin_to_coords[bin_num] = (chrom, start, end)
                
                # Store bin numbers for this chromosome
                if chrom not in chrom_bins:
                    chrom_bins[chrom] = []
                chrom_bins[chrom].append((start, end, bin_num - 1))  # Store 0-indexed
                
                # Map genomic positions to bins
                for pos in range(start // resolution, end // resolution):
                    coord_to_bin[pos] = bin_num - 1
    
    return bin_to_coords, coord_to_bin, chrom_bins

def read_hic_matrix(filepath, bin_mapping=None, resolution=20000):
    """Read Hi-C matrix in sparse format"""
    print(f"Reading {filepath}...")
    
    data = []
    row_indices = []
    col_indices = []
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                try:
                    i = int(parts[0]) - 1
                    j = int(parts[1]) - 1
                    value = float(parts[2])
                    
                    row_indices.append(i)
                    col_indices.append(j)
                    data.append(value)
                    
                    if i != j:
                        row_indices.append(j)
                        col_indices.append(i)
                        data.append(value)
                except (ValueError, IndexError):
                    continue
    
    if not data:
        print(f"Warning: No data found in {filepath}")
        return None, 0
    
    if bin_mapping and bin_mapping[0]:
        max_idx = len(bin_mapping[0])
    else:
        max_idx = max(max(row_indices), max(col_indices)) + 1
    
    sparse_matrix = coo_matrix((data, (row_indices, col_indices)), 
                               shape=(max_idx, max_idx))
    dense_matrix = sparse_matrix.toarray()
    total_contacts = np.sum(data) / 2
    
    print(f"  Matrix shape: {dense_matrix.shape}")
    print(f"  Non-zero entries: {len(data)//2}")
    print(f"  Total contacts: {total_contacts:.0f}")
    
    return dense_matrix, total_contacts

def calculate_expected(matrix, chrom_boundaries=None):
    """
    Calculate expected values based on genomic distance
    Returns expected matrix with same shape
    Chromosome-aware version for post-PDE samples
    
    Parameters:
    -----------
    matrix : ndarray
        Hi-C contact matrix
    chrom_boundaries : list of int, optional
        List of bin indices where chromosomes start (for post-PDE)
        If provided, expected is calculated per-chromosome for cis,
        and uses trans average for inter-chromosomal regions
    """
    if matrix is None:
        return None
    
    n = matrix.shape[0]
    expected = np.zeros_like(matrix, dtype=np.float32)
    
    if chrom_boundaries is None or len(chrom_boundaries) <= 1:
        # Original behavior: treat as single chromosome
        for d in range(n):
            diag_indices = np.arange(n - d)
            diag_values = matrix[diag_indices, diag_indices + d]
            valid_values = diag_values[diag_values > 0]
            if len(valid_values) > 0:
                mean_val = np.mean(valid_values)
            else:
                mean_val = 0
            expected[diag_indices, diag_indices + d] = mean_val
            expected[diag_indices + d, diag_indices] = mean_val
    else:
        # Chromosome-aware calculation
        # Add end boundary
        boundaries = [0] + sorted(chrom_boundaries) + [n]
        
        # Calculate per-chromosome expected (cis)
        for i in range(len(boundaries) - 1):
            chrom_start = boundaries[i]
            chrom_end = boundaries[i + 1]
            chrom_size = chrom_end - chrom_start
            
            # Calculate expected for this chromosome
            for d in range(chrom_size):
                diag_indices = np.arange(chrom_start, chrom_end - d)
                diag_values = matrix[diag_indices, diag_indices + d]
                valid_values = diag_values[diag_values > 0]
                
                if len(valid_values) > 0:
                    mean_val = np.mean(valid_values)
                else:
                    mean_val = 0
                
                expected[diag_indices, diag_indices + d] = mean_val
                expected[diag_indices + d, diag_indices] = mean_val
        
        # Calculate trans expected (inter-chromosomal)
        trans_values = []
        for i in range(len(boundaries) - 1):
            for j in range(i + 1, len(boundaries) - 1):
                chrom1_start = boundaries[i]
                chrom1_end = boundaries[i + 1]
                chrom2_start = boundaries[j]
                chrom2_end = boundaries[j + 1]
                
                # Get all trans contacts between these chromosomes
                trans_block = matrix[chrom1_start:chrom1_end, chrom2_start:chrom2_end]
                trans_values.extend(trans_block[trans_block > 0].flatten())
        
        # Set trans expected
        if len(trans_values) > 0:
            trans_mean = np.mean(trans_values)
        else:
            trans_mean = 0
        
        for i in range(len(boundaries) - 1):
            for j in range(i + 1, len(boundaries) - 1):
                chrom1_start = boundaries[i]
                chrom1_end = boundaries[i + 1]
                chrom2_start = boundaries[j]
                chrom2_end = boundaries[j + 1]
                
                expected[chrom1_start:chrom1_end, chrom2_start:chrom2_end] = trans_mean
                expected[chrom2_start:chrom2_end, chrom1_start:chrom1_end] = trans_mean
    
    return expected

def calculate_obs_exp(matrix, use_log2=False, chrom_boundaries=None):
    """
    Calculate observed/expected matrix, optionally log2 transformed
    
    Parameters:
    -----------
    matrix : ndarray
        Hi-C contact matrix
    use_log2 : bool
        Apply log2 transformation
    chrom_boundaries : list of int, optional
        Chromosome boundaries for post-PDE samples
    """
    if matrix is None:
        return None
    
    print("  Calculating expected values...")
    if chrom_boundaries is not None:
        print(f"  Using chromosome-aware expected (boundaries at bins: {chrom_boundaries})")
    expected = calculate_expected(matrix, chrom_boundaries=chrom_boundaries)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        obs_exp = np.divide(matrix, expected, dtype=np.float32)
        obs_exp[~np.isfinite(obs_exp)] = 0
    
    # Clean up expected matrix
    del expected
    
    # Apply log2 transformation if requested
    if use_log2:
        print("  Applying log2 transformation...")
        with np.errstate(divide='ignore', invalid='ignore'):
            obs_exp = np.log2(obs_exp, dtype=np.float32)
            obs_exp[~np.isfinite(obs_exp)] = 0
    
    return obs_exp

def extract_region_germline(matrix, start_coord, end_coord, coord_to_bin=None, resolution=20000):
    """Extract a specific genomic region from the matrix (for germline/pre-PDE)"""
    if matrix is None:
        return None
    
    if coord_to_bin:
        start_bin_key = start_coord // resolution
        end_bin_key = end_coord // resolution
        start_bin = coord_to_bin.get(start_bin_key, 0)
        end_bin = coord_to_bin.get(end_bin_key, matrix.shape[0])
    else:
        start_bin = start_coord // resolution
        end_bin = end_coord // resolution
    
    max_bin = matrix.shape[0]
    start_bin = max(0, min(start_bin, max_bin))
    end_bin = max(0, min(end_bin, max_bin))
    
    if start_bin >= end_bin:
        return None
    
    return matrix[start_bin:end_bin, start_bin:end_bin]

def extract_region_somatic(matrix, soma_chrom_range, chrom_bins):
    """
    Extract a specific genomic region from the matrix (for somatic/post-PDE)
    Returns: (submatrix, chrom_boundaries_in_submatrix)
    """
    if matrix is None or soma_chrom_range is None or chrom_bins is None:
        return None, None
    
    # Collect all bins for the specified somatic chromosome range
    chrom_bin_ranges = []  # List of (chrom_name, bins) for each chromosome
    
    for chrom, start, end in soma_chrom_range:
        if chrom in chrom_bins:
            # Get all bins for this chromosome
            chrom_bins_list = []
            for bin_start, bin_end, bin_idx in sorted(chrom_bins[chrom]):
                chrom_bins_list.append(bin_idx)
            if chrom_bins_list:
                chrom_bin_ranges.append((chrom, chrom_bins_list))
    
    if not chrom_bin_ranges:
        print(f"  Warning: No bins found for somatic region")
        return None, None
    
    # Flatten all bins
    all_bins = []
    for chrom, bins in chrom_bin_ranges:
        all_bins.extend(bins)
    
    # Get min and max bin indices
    min_bin = min(all_bins)
    max_bin = max(all_bins) + 1
    
    # Extract the submatrix
    if min_bin >= max_bin or min_bin >= matrix.shape[0] or max_bin > matrix.shape[0]:
        print(f"  Warning: Invalid bin range {min_bin}:{max_bin} for matrix shape {matrix.shape}")
        return None, None
    
    submatrix = matrix[min_bin:max_bin, min_bin:max_bin]
    print(f"  Extracted somatic region: bins {min_bin}:{max_bin}, shape {submatrix.shape}")
    
    # Calculate chromosome boundaries within the submatrix
    chrom_boundaries = []
    current_pos = 0
    for i, (chrom, bins) in enumerate(chrom_bin_ranges):
        if i > 0:  # First chromosome starts at 0, so only add boundaries for subsequent ones
            chrom_boundaries.append(current_pos)
        current_pos += len(bins)
        print(f"  Chromosome {chrom}: {len(bins)} bins, boundary at {current_pos if i < len(chrom_bin_ranges)-1 else 'end'}")
    
    return submatrix, chrom_boundaries

def plot_upper_triangle_compact(ax, matrix, y_offset, height, label, vmax, vmin_threshold=0.1,
                                normalize_factor=1.0, colormap='YlOrRd', add_outline=True):
    """Plot upper triangular Hi-C matrix pointing upward with optional outline"""
    if matrix is None:
        ax.text(0.5, y_offset + height/2, f'{label}\n(No data)', 
                ha='center', va='center', fontsize=10, style='italic', family='arial')
        return
    
    n = matrix.shape[0]
    
    # Normalize matrix
    matrix_plot = matrix.copy() * normalize_factor
    matrix_plot[np.isnan(matrix_plot)] = 0
    
    vmax_scaled = vmax
    vmin_scaled = vmin_threshold
    
    # Get colormap
    cmap = plt.colormaps.get_cmap(colormap)
    
    # Plot upper triangle
    for i in range(n):
        for j in range(i, n):
            if matrix_plot[i, j] > vmin_scaled:
                x = (i + j) / 2.0 / n
                y = y_offset + (j - i) / n * height
                
                if y <= y_offset + height:
                    w = 1.0 / n
                    h = height / n
                    
                    color_intensity = min((matrix_plot[i, j] - vmin_scaled) / (vmax_scaled - vmin_scaled), 1.0)
                    color_intensity = max(0, color_intensity)
                    
                    points = [
                        [x - w/2, y],
                        [x, y + h/2],
                        [x + w/2, y],
                        [x, y - h/2]
                    ]
                    
                    color = cmap(color_intensity)
                    polygon = Polygon(points, facecolor=color, edgecolor='none', 
                                    linewidth=0, antialiased=True)
                    ax.add_patch(polygon)
    
    # Add label with larger font
    ax.text(-0.08, y_offset + height/2, label, 
            ha='right', va='center', fontsize=4, fontweight='bold', family='arial')
    
    # Add triangle outline (very thin)
    if add_outline:
        border_points = [
            [0, y_offset],
            [0.5, y_offset + height],
            [1, y_offset],
            [0, y_offset]
        ]
        border = Polygon(border_points, facecolor='none', 
                        edgecolor='black', linewidth=0.15, alpha=1.0)
        ax.add_patch(border)

def create_stacked_hic_plot(matrices_data, start_coord, end_coord, 
                           vmax=100, vmin_threshold=0.1, output_prefix='hic_stages', 
                           resolution=20000, coord_to_bin=None, normalize=True,
                           plot_type='raw', add_outline=True, vmax_obsexp=3.0, vmin_obsexp=0.5,
                           use_log2_obsexp=False, output_dir='parascaris_triangle_plots',
                           soma_chrom_range=None, soma_chrom_bins=None):
    """Create stacked Hi-C plot for multiple developmental stages"""
    
    # Stage name mapping for display (keep original for filenames)
    stage_display_names = {
        '10hr': '1-2 cells',
        '13.5hr': '13.5hr',
        '17hr': '2-4 cell',
        '24hr': '4-8 cell',
        '36hr': '36hr',
        '48hr': 'late embryo',
        '60hr': '60hr',
        '72hr': 'L1',
        'Testis': 'Testis',
        'Ovary': 'Ovary',
        'Female Intestine': 'Female Intestine',
        'Male Intestine': 'Male Intestine'
    }
    
    # Determine colormap and scale based on plot type
    if plot_type == 'obsexp':
        colormap = 'RdBu_r'
        if use_log2_obsexp:
            # For log2(obs/exp), use symmetric range around 0
            vmax = vmax_obsexp  # e.g., 2 means -2 to +2
            vmin_threshold = -vmax_obsexp
            scale_label = 'log₂(Obs/Exp)'
        else:
            vmax = vmax_obsexp
            vmin_threshold = vmin_obsexp
            scale_label = 'Obs/Exp'
    else:
        colormap = 'YlOrRd'
        scale_label = 'Contact Frequency'
    
    matrices_data = list(reversed(matrices_data))
    
    n_stages = len(matrices_data)
    fig_height = 0.6 * n_stages + 1
    fig_width = 8
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    plot_height = 0.5 * n_stages
    plot_fraction = plot_height / fig_height
    
    ax = fig.add_axes([0.15, 0.15, 0.65, plot_fraction])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_stages * 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Calculate normalization factors
    if normalize and plot_type != 'obsexp':
        total_contacts_list = [tc for _, _, tc, _ in matrices_data if tc > 0]
        if total_contacts_list:
            median_contacts = np.median(total_contacts_list)
            print(f"\nMedian total contacts for normalization: {median_contacts:.0f}")
        else:
            median_contacts = 1.0
    else:
        median_contacts = 1.0
    
    triangle_height = 0.5
    vertical_spacing = 0.5
    
    for idx, (stage_name, matrix, total_contacts, is_postpde) in enumerate(matrices_data):
        y_offset = idx * vertical_spacing
        
        # Get display name for this stage
        display_name = stage_display_names.get(stage_name, stage_name)
        
        # Calculate normalization factor
        if normalize and total_contacts > 0 and plot_type != 'obsexp':
            normalize_factor = median_contacts / total_contacts
            print(f"{stage_name}: Normalization factor = {normalize_factor:.3f}")
        else:
            normalize_factor = 1.0
        
        # Extract region based on whether it's pre- or post-PDE
        chrom_boundaries = None  # For obs/exp calculation
        if matrix is not None:
            if is_postpde and soma_chrom_range is not None and soma_chrom_bins is not None:
                # Post-PDE: use somatic chromosome extraction
                print(f"  Extracting somatic region for {stage_name} (post-PDE)")
                region_matrix, chrom_boundaries = extract_region_somatic(matrix, soma_chrom_range, soma_chrom_bins)
            else:
                # Pre-PDE: use germline coordinate extraction
                region_matrix = extract_region_germline(matrix, start_coord, end_coord, 
                                                       coord_to_bin, resolution)
            
            # Calculate obs/exp if requested
            if plot_type == 'obsexp' and region_matrix is not None:
                region_matrix = calculate_obs_exp(region_matrix, use_log2=use_log2_obsexp, 
                                                 chrom_boundaries=chrom_boundaries)
                normalize_factor = 1.0  # Don't normalize obs/exp
                gc.collect()  # Free memory after obs/exp calculation
        else:
            region_matrix = None
        
        # Plot triangle with display name
        plot_upper_triangle_compact(ax, region_matrix, y_offset, 
                          triangle_height, display_name, vmax, 
                          vmin_threshold, normalize_factor, 
                          colormap=colormap, add_outline=add_outline)
        
        # Add horizontal separator
        if idx < n_stages - 1:
            ax.axhline(y=y_offset + vertical_spacing, color='black', linewidth=0.5, 
                      xmin=-0.05, xmax=1.05, alpha=0.7, clip_on=False)
    
    # Add genomic coordinates with larger font
    region_size_mb = (end_coord - start_coord) / 1e6
    n_ticks = min(10, int(region_size_mb) + 1)  # Limit number of ticks for clarity
    
    for i in range(n_ticks):
        mb_value = int(start_coord/1e6) + i * int(region_size_mb / (n_ticks - 1))
        pos = i / (n_ticks - 1)
        if pos <= 1.0:
            ax.text(pos, -0.03, f'{mb_value}', ha='center', va='top', 
                   fontsize=6, family='arial', transform=ax.transData)
    
    ax.text(0.5, -0.08, 'Genomic Position (Mb)', ha='center', va='top', 
           fontsize=10, family='arial', fontweight='bold', transform=ax.transData)
    
    # Bottom line
    ax.axhline(y=0, color='black', linewidth=0.5, xmin=-0.05, xmax=1.05, alpha=0.7, clip_on=False)
    
    # Title with larger font
    norm_text = " (Normalized)" if normalize and plot_type != 'obsexp' else ""
    type_text = " - Obs/Exp" if plot_type == 'obsexp' and not use_log2_obsexp else ""
    type_text = " - log₂(Obs/Exp)" if plot_type == 'obsexp' and use_log2_obsexp else type_text
    title_y = n_stages * 0.5 + 0.05
    
    if plot_type == 'obsexp' and use_log2_obsexp:
        range_text = f'Range: {vmin_threshold:.1f} to {vmax:.1f}'
    else:
        range_text = f'Max: {vmax}, Min threshold: {vmin_threshold}'
    
    ax.text(0.5, title_y, 
           f'Hi-C Matrices{type_text}: chrX:{start_coord:,}-{end_coord:,} ({region_size_mb:.1f} Mb){norm_text}\n' + 
           f'Resolution: {resolution//1000}kb, {range_text}',
           ha='center', va='bottom', fontsize=14, fontweight='bold', family='arial',
           transform=ax.transData)
    
    # Colorbar with larger label
    cbar_height = min(0.6, plot_fraction * 0.8)
    cbar_bottom = 0.15 + (plot_fraction - cbar_height) / 2
    cbar_ax = fig.add_axes([0.85, cbar_bottom, 0.02, cbar_height])
    
    if plot_type == 'obsexp' and use_log2_obsexp:
        # Symmetric colormap for log2(obs/exp) centered at 0
        norm = mcolors.TwoSlopeNorm(vmin=vmin_threshold, vcenter=0.0, vmax=vmax)
    elif plot_type == 'obsexp':
        # Diverging colormap for obs/exp centered at 1
        norm = mcolors.TwoSlopeNorm(vmin=vmin_threshold, vcenter=1.0, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin_threshold, vmax=vmax)
    
    cmap = plt.colormaps.get_cmap(colormap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label(scale_label, fontsize=12, family='arial', fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    for label in cbar.ax.get_yticklabels():
        label.set_family('arial')
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    
    n_bins = (end_coord - start_coord) // resolution
    data_dpi = max(1200, int(n_bins / 4))
    
    if plot_type == 'obsexp' and use_log2_obsexp:
        type_suffix = '_log2obsexp'
    elif plot_type == 'obsexp':
        type_suffix = '_obsexp'
    else:
        type_suffix = ''
    
    print(f"\nSaving figures to {output_dir}/...")
    
    # Save high-res PNG
    png_path = f'{output_dir}/{output_prefix}{type_suffix}_{start_coord}_{end_coord}.png'
    plt.savefig(png_path, dpi=data_dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved PNG: {png_path} at {data_dpi} DPI")
    
    # Save TIFF
    tiff_path = f'{output_dir}/{output_prefix}{type_suffix}_{start_coord}_{end_coord}.tiff'
    plt.savefig(tiff_path, format='tiff', dpi=data_dpi, bbox_inches='tight', 
                facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
    print(f"Saved TIFF: {tiff_path} at {data_dpi} DPI")
    
    # Save SVG
    svg_path = f'{output_dir}/{output_prefix}{type_suffix}_{start_coord}_{end_coord}.svg'
    plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    print(f"Saved SVG: {svg_path}")
    
    # Report file sizes
    if os.path.exists(png_path):
        png_size = os.path.getsize(png_path) / (1024*1024)
        print(f"PNG file size: {png_size:.1f} MB")
    if os.path.exists(tiff_path):
        tiff_size = os.path.getsize(tiff_path) / (1024*1024)
        print(f"TIFF file size: {tiff_size:.1f} MB")
    if os.path.exists(svg_path):
        svg_size = os.path.getsize(svg_path) / (1024*1024)
        print(f"SVG file size: {svg_size:.1f} MB")
    
    plt.close()
    gc.collect()  # Free memory after saving plots

def main():
    parser = argparse.ArgumentParser(description='Enhanced Hi-C visualization for P. univalens')
    parser.add_argument('--region', type=str, default='10000000-15000000',
                       help='Region to plot: "chr03-chr08" or "10000000-15000000"')
    parser.add_argument('--mapping', type=str, 
                       default='germ_to_soma_mapping.bed',
                       help='Path to germline-to-somatic chromosome mapping file')
    parser.add_argument('--vmax', type=float, default=100,
                       help='Maximum value for color scaling (raw data)')
    parser.add_argument('--vmin', type=float, default=0.1,
                       help='Minimum threshold (raw data)')
    parser.add_argument('--vmax-obsexp', type=float, default=3.0,
                       help='Maximum value for obs/exp plots (default: 3.0)')
    parser.add_argument('--vmin-obsexp', type=float, default=0.5,
                       help='Minimum threshold for obs/exp plots (default: 0.5)')
    parser.add_argument('--resolution', type=int, default=20000,
                       help='Hi-C matrix resolution in bp (default: 20000)')
    parser.add_argument('--timepoints', type=str, default='10hr,17hr,24hr,48hr,72hr',
                       help='Comma-separated list of timepoints')
    parser.add_argument('--bin-bed', type=str, default='20000/data1_20000_abs.bed',
                       help='Path to bin mapping bed file (germline genome)')
    parser.add_argument('--soma-bin-bed', type=str, default='',
                       help='Path to somatic bin mapping bed file (for post-PDE samples)')
    parser.add_argument('--no-normalize', action='store_true',
                       help='Do not normalize matrices by total contacts')
    parser.add_argument('--plot-type', type=str, default='both', 
                       choices=['raw', 'obsexp', 'both'],
                       help='Type of plot: raw, obsexp, or both (default: both)')
    parser.add_argument('--log2-obsexp', action='store_true',
                       help='Use log2 transformation for obs/exp (symmetric scale around 0)')
    parser.add_argument('--no-outline', action='store_true',
                       help='Do not add black outline to triangles')
    parser.add_argument('--output-dir', type=str, default='parascaris_triangle_plots',
                       help='Output directory for plots (default: parascaris_triangle_plots)')
    
    args = parser.parse_args()
    
    # Expand home directory in mapping path
    args.mapping = os.path.expanduser(args.mapping)
    
    print("="*60)
    print("Hi-C Triangle Visualization for Parascaris univalens")
    print("="*60)
    
    # Parse region
    try:
        start_coord, end_coord = parse_region_arg(args.region, args.mapping)
        print(f"Region: chrX:{start_coord:,}-{end_coord:,}")
    except Exception as e:
        print(f"Error parsing region: {e}")
        sys.exit(1)
    
    # Read chromosome mapping to determine somatic chromosomes
    chr_mappings = read_chromosome_mapping(args.mapping)
    
    # Determine which somatic chromosomes correspond to this region
    soma_chrom_range = None
    if '-' in args.region and args.region.split('-')[0].startswith('chr'):
        # User specified chromosome range like "chrX7-chrX9"
        chr_parts = args.region.split('-')
        soma_chr_start = chr_parts[0]
        soma_chr_end = chr_parts[1]
        soma_chrom_range = get_somatic_region(soma_chr_start, soma_chr_end, chr_mappings)
        print(f"Somatic chromosomes: {soma_chr_start} to {soma_chr_end}")
        if soma_chrom_range:
            for chrom, start, end in soma_chrom_range:
                print(f"  {chrom}: {start:,}-{end:,}")
    
    print(f"Resolution: {args.resolution} bp")
    print(f"Plot type: {args.plot_type}")
    print(f"Outline: {'OFF' if args.no_outline else 'ON'}")
    print("="*60)
    
    # Read bin mapping (germline)
    bin_to_coords = None
    coord_to_bin = None
    chrom_bins_germ = None
    if os.path.exists(args.bin_bed):
        print(f"\nReading germline bin mapping from {args.bin_bed}...")
        bin_to_coords, coord_to_bin, chrom_bins_germ = read_bin_mapping(args.bin_bed, args.resolution)
        print(f"Loaded mapping for {len(bin_to_coords)} bins")
        print(f"Chromosomes in germline mapping: {list(chrom_bins_germ.keys())[:5]}...")
    
    # Read somatic bin mapping (for post-PDE)
    soma_bin_to_coords = None
    soma_coord_to_bin = None
    soma_chrom_bins = None
    if args.soma_bin_bed and os.path.exists(args.soma_bin_bed):
        print(f"\nReading somatic bin mapping from {args.soma_bin_bed}...")
        soma_bin_to_coords, soma_coord_to_bin, soma_chrom_bins = read_bin_mapping(args.soma_bin_bed, args.resolution)
        print(f"Loaded somatic mapping for {len(soma_bin_to_coords)} bins")
        print(f"Chromosomes in somatic mapping: {list(soma_chrom_bins.keys())[:5]}...")
        
        # Debug: Check if our target chromosomes are in the mapping
        if soma_chrom_range:
            for chrom, _, _ in soma_chrom_range:
                if chrom in soma_chrom_bins:
                    print(f"  {chrom}: {len(soma_chrom_bins[chrom])} bins")
                else:
                    print(f"  WARNING: {chrom} not found in somatic bin mapping!")
    elif soma_chrom_range is not None:
        print(f"\nWARNING: Somatic region specified but no --soma-bin-bed provided")
        print(f"Post-PDE samples may not extract correctly")
    
    # Define stages
    stages_prepde = [
        ('10hr', f'matrix_files_{args.resolution//1000}kb/prepde/pu_10hr_{args.resolution//1000}kb_iced.matrix'),
        ('13.5hr', f'matrix_files_{args.resolution//1000}kb/prepde/pu_13.5hr_{args.resolution//1000}kb_iced.matrix'),
        ('17hr', f'matrix_files_{args.resolution//1000}kb/prepde/pu_17hr_{args.resolution//1000}kb_iced.matrix'),
    ]
    
    stages_postpde = [
        ('24hr', f'matrix_files_{args.resolution//1000}kb/postpde/pu_24hr_{args.resolution//1000}kb_iced.matrix'),
        ('36hr', f'matrix_files_{args.resolution//1000}kb/postpde/pu_36hr_{args.resolution//1000}kb_iced.matrix'),
        ('48hr', f'matrix_files_{args.resolution//1000}kb/postpde/pu_48hr_{args.resolution//1000}kb_iced.matrix'),
        ('60hr', f'matrix_files_{args.resolution//1000}kb/postpde/pu_60hr_{args.resolution//1000}kb_iced.matrix'),
        ('72hr', f'matrix_files_{args.resolution//1000}kb/postpde/pu_72hr_{args.resolution//1000}kb_iced.matrix'),
    ]
    
    all_stages = stages_prepde + stages_postpde
    
    # Filter timepoints
    if args.timepoints != 'all':
        selected_timepoints = [s.strip() for s in args.timepoints.split(',')]
        all_stages = [(name, path) for name, path in all_stages 
                     if name in selected_timepoints]
    
    # Load matrices
    print("\nLoading Hi-C matrices...")
    matrices_data = []
    
    for stage_name, filepath in all_stages:
        is_postpde = 'postpde' in filepath
        
        if os.path.exists(filepath):
            # Use appropriate bin mapping for pre/post PDE
            if is_postpde and soma_bin_to_coords is not None:
                print(f"Loading {stage_name} with somatic bin mapping...")
                matrix, total_contacts = read_hic_matrix(filepath, (soma_bin_to_coords, soma_coord_to_bin), args.resolution)
            else:
                matrix, total_contacts = read_hic_matrix(filepath, (bin_to_coords, coord_to_bin), args.resolution)
            matrices_data.append((stage_name, matrix, total_contacts, is_postpde))
        else:
            print(f"Warning: File not found: {filepath}")
            matrices_data.append((stage_name, None, 0, is_postpde))
    
    if not matrices_data:
        print("Error: No matrices loaded!")
        sys.exit(1)
    
    # Create plots
    plot_types = ['raw', 'obsexp'] if args.plot_type == 'both' else [args.plot_type]
    
    for ptype in plot_types:
        print(f"\nCreating {ptype} plot...")
        create_stacked_hic_plot(matrices_data, start_coord, end_coord, 
                               args.vmax, args.vmin, 'hic_developmental', 
                               args.resolution, coord_to_bin, 
                               normalize=(not args.no_normalize),
                               plot_type=ptype,
                               add_outline=(not args.no_outline),
                               vmax_obsexp=args.vmax_obsexp,
                               vmin_obsexp=args.vmin_obsexp,
                               use_log2_obsexp=args.log2_obsexp,
                               output_dir=args.output_dir,
                               soma_chrom_range=soma_chrom_range,
                               soma_chrom_bins=soma_chrom_bins)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)

if __name__ == '__main__':
    main()