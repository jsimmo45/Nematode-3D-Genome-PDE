#!/usr/bin/env python3
"""
plot_ev1_multirow.py
====================
Compare Hi-C compartment eigenvector 1 (EV1) values between Ascaris and
Parascaris across their shared germline chromosome, displayed in a multi-row
layout.  Each row contains a paired Ascaris (top) and Parascaris (bottom) panel
spanning a user-defined set of somatic chromosomes, with eliminated DNA regions
shaded in red and per-chromosome Pearson correlations annotated.

This script handles the coordinate system complexities of comparing two species
that undergo programmed DNA elimination: Parascaris post-PDE somatic
eigenvectors are mapped back to germline coordinates, and Ascaris chromosomes
are mapped to their Parascaris orthologs (with orientation reversal and EV1
sign-flipping as needed) so both species are displayed on the same germline
coordinate axis.

Outputs:
  - Multi-row PNG at configurable DPI
  - SVG with editable text for Adobe Illustrator

Dependencies:
  numpy, pandas, matplotlib, seaborn, scipy

Example usage:
  python plot_ev1_multirow.py \\
      eigenvectors/pu_72hr_iced_100kb_ev1.matrix.eigenvector \\
      --as-eigenvector eigenvectors/as_10day_iced_100kb_ev1.matrix.eigenvector \\
      --elimination-bed data/pu_v2_eliminated.bed \\
      --as-elimination-bed data/AG_v50_eliminated_strict.bed \\
      --germ-soma-mapping data/pu_v2_germ_to_soma_mapping.bed \\
      --as-pu-mapping data/as_to_pu_chrom_order.txt \\
      --flip-file data/flip_or_not.txt \\
      --chromosome chrX \\
      --row-chromosomes "chr01-chr08;chrX1-chrX9;chr09-chr16;chr17-chr27" \\
      --smooth --window 3 \\
      --y-min -0.3 --y-max 0.3 \\
      --linewidth 2.5 \\
      --pu-color "#0066CC" --as-color "red" \\
      --output multirow_comparison \\
      --dpi 300

Row definition (--row-chromosomes):
  Rows are separated by semicolons (;).  Within a row, use dashes for ranges
  (chr01-chr08) or commas for explicit lists (chr01,chr02,chr03).  Each row
  is plotted with width proportional to its genomic span.

Input files:
  Positional arg: Parascaris post-PDE eigenvector file (FANC format:
      chr  start  end  eigenvector).
  --as-eigenvector: Ascaris eigenvector file (same format).
  --elimination-bed: BED file of Parascaris eliminated DNA regions.
  --as-elimination-bed: BED file of Ascaris eliminated DNA regions.
  --germ-soma-mapping: Germline-to-somatic coordinate mapping
      (germ_chr  start  end  soma_chr).
  --as-pu-mapping: Ascaris-to-Parascaris chromosome orthology mapping
      (ascaris_chr  parascaris_chr  orientation).
  --flip-file: Per-chromosome EV1 sign correction file (4-column:
      pu_chr  flip/correct  as_chr_short  flip/correct).
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats

# Set publication quality parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.labelsize'] = 16  # Bigger axis labels
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8

def read_bed_file(bed_file):
    """Read BED file and return regions as dataframe."""
    try:
        bed_df = pd.read_csv(bed_file, sep='\t', header=None)
        bed_df.columns = ['chr', 'start', 'end'] + [f'col{i}' for i in range(3, len(bed_df.columns))]
        bed_df = bed_df[['chr', 'start', 'end']]
        print(f"  Loaded {len(bed_df)} regions from {bed_file}")
        return bed_df
    except Exception as e:
        print(f"Warning: Could not load BED file {bed_file}: {e}")
        return pd.DataFrame(columns=['chr', 'start', 'end'])

def read_eigenvector_file(ev_file):
    """Read FANC format eigenvector file."""
    try:
        ev_df = pd.read_csv(ev_file, sep='\t', header=None,
                           names=['chr', 'start', 'end', 'eigenvector'])
        print(f"  Loaded {len(ev_df)} bins from {ev_file}")
        return ev_df
    except Exception as e:
        print(f"Error loading eigenvector file {ev_file}: {e}")
        return None

def read_germ_soma_mapping(mapping_file):
    """Read germ-to-soma mapping BED file."""
    try:
        mapping_df = pd.read_csv(mapping_file, sep='\t', header=None,
                                names=['germ_chr', 'germ_start', 'germ_end', 'soma_chr'])
        print(f"  Loaded {len(mapping_df)} soma->germ mappings from {mapping_file}")
        return mapping_df
    except Exception as e:
        print(f"Warning: Could not load germ-soma mapping file {mapping_file}: {e}")
        return pd.DataFrame()

def read_chromosome_mapping(mapping_file):
    """Read chromosome mapping file for Ascaris->Parascaris."""
    try:
        mapping_df = pd.read_csv(mapping_file, sep='\t', header=None,
                                names=['ascaris_chr', 'parascaris_chr', 'orientation'])
        print(f"  Loaded {len(mapping_df)} Ascaris->Parascaris mappings from {mapping_file}")
        return mapping_df
    except Exception as e:
        print(f"Warning: Could not load mapping file {mapping_file}: {e}")
        return pd.DataFrame()

def read_flip_file(flip_file):
    """Read flip file with 4-column format."""
    parascaris_flip = {}
    ascaris_flip = {}
    try:
        with open(flip_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 4:
                        pu_chr = parts[0]
                        pu_flip = parts[1].lower() == 'flip'
                        as_chr = parts[2]
                        as_flip = parts[3].lower() == 'flip'
                        
                        parascaris_flip[pu_chr] = pu_flip
                        if not pu_chr.startswith('chr'):
                            parascaris_flip['chr' + pu_chr] = pu_flip
                        
                        ascaris_flip[as_chr] = as_flip
                        if as_chr.startswith('chr'):
                            ascaris_flip[as_chr[3:]] = as_flip
                        else:
                            ascaris_flip['chr' + as_chr] = as_flip
        
        print(f"  Loaded flip information: {len(parascaris_flip)//2} Parascaris, {len(ascaris_flip)//2} Ascaris chromosomes")
        return parascaris_flip, ascaris_flip
    except Exception as e:
        print(f"Warning: Could not load flip file {flip_file}: {e}")
        return {}, {}

def map_soma_to_germ_coordinates(soma_ev_data, germ_soma_mapping, parascaris_flip, target_germ_chr='chrX'):
    """Map post-PDE soma eigenvector data to pre-PDE germ coordinates."""
    if soma_ev_data is None or germ_soma_mapping.empty:
        return None
    
    mapped_data = []
    germ_mappings = germ_soma_mapping[germ_soma_mapping['germ_chr'] == target_germ_chr]
    
    if germ_mappings.empty:
        print(f"  No mappings found for {target_germ_chr}")
        return None
    
    print(f"  Mapping {len(germ_mappings)} soma chromosomes to {target_germ_chr}")
    
    for _, mapping in germ_mappings.iterrows():
        soma_chr = mapping['soma_chr']
        germ_start = mapping['germ_start']
        germ_end = mapping['germ_end']
        
        soma_chr_data = soma_ev_data[soma_ev_data['chr'] == soma_chr]
        
        if soma_chr_data.empty:
            continue
        
        should_flip = parascaris_flip.get(soma_chr, False)
        soma_length = soma_chr_data['end'].max() - soma_chr_data['start'].min()
        germ_length = germ_end - germ_start
        scale_factor = germ_length / soma_length if soma_length > 0 else 1
        
        for _, soma_bin in soma_chr_data.iterrows():
            soma_pos = (soma_bin['start'] + soma_bin['end']) / 2
            soma_relative_pos = soma_pos - soma_chr_data['start'].min()
            germ_pos = germ_start + (soma_relative_pos * scale_factor)
            eigenvector_value = -soma_bin['eigenvector'] if should_flip else soma_bin['eigenvector']
            
            mapped_data.append({
                'position': germ_pos,
                'eigenvector': eigenvector_value,
                'original_chr': soma_chr,
                'germ_chr': target_germ_chr,
                'flipped': should_flip
            })
    
    if mapped_data:
        mapped_df = pd.DataFrame(mapped_data).sort_values('position')
        return mapped_df
    return None

def map_ascaris_to_prepde_coordinates(as_ev_data, as_pu_mapping, germ_soma_mapping, ascaris_flip, target_germ_chr='chrX'):
    """Map Ascaris data to pre-PDE coordinates."""
    if as_ev_data is None or as_pu_mapping.empty or germ_soma_mapping.empty:
        return None
    
    mapped_data = []
    germ_mappings = germ_soma_mapping[germ_soma_mapping['germ_chr'] == target_germ_chr]
    
    if germ_mappings.empty:
        return None
    
    for _, germ_mapping in germ_mappings.iterrows():
        postpde_chr = germ_mapping['soma_chr']
        germ_start = germ_mapping['germ_start']
        germ_end = germ_mapping['germ_end']
        germ_length = germ_end - germ_start
        
        as_mappings = as_pu_mapping[as_pu_mapping['parascaris_chr'] == postpde_chr]
        
        for _, as_mapping in as_mappings.iterrows():
            as_chr = as_mapping['ascaris_chr']
            orientation = as_mapping['orientation']
            
            as_chr_data = as_ev_data[as_ev_data['chr'] == as_chr]
            if as_chr_data.empty:
                continue
            
            as_chr_data_sorted = as_chr_data.sort_values('start').copy()
            as_positions = (as_chr_data_sorted['start'] + as_chr_data_sorted['end']) / 2
            as_values = as_chr_data_sorted['eigenvector'].values.copy()
            
            should_flip_as = False
            for chr_format in [as_chr, as_chr.replace('chr', ''), 'chr' + as_chr]:
                if chr_format in ascaris_flip:
                    should_flip_as = ascaris_flip[chr_format]
                    break
            
            if should_flip_as:
                as_values = -as_values
            
            as_pos_norm = (as_positions - as_positions.min()) / (as_positions.max() - as_positions.min())
            
            if orientation == 'reverse':
                as_pos_norm = 1 - as_pos_norm
            
            mapped_positions = germ_start + (as_pos_norm * germ_length)
            
            for pos, val in zip(mapped_positions, as_values):
                mapped_data.append({
                    'position': pos,
                    'eigenvector': val,
                    'original_chr': as_chr,
                    'orientation': orientation,
                    'via_postpde': postpde_chr
                })
    
    if mapped_data:
        return pd.DataFrame(mapped_data).sort_values('position')
    return None

def smooth_data(data, window_size=5):
    """Apply rolling mean smoothing."""
    if len(data) < window_size:
        return data
    return data.rolling(window=window_size, center=True, min_periods=1).mean()

def mask_eliminated_regions(positions, values, elimination_regions):
    """Remove data points in eliminated regions."""
    if elimination_regions.empty or len(positions) == 0:
        return positions, values
    
    keep_mask = np.ones(len(positions), dtype=bool)
    
    for i, pos in enumerate(positions):
        for _, elim_region in elimination_regions.iterrows():
            if elim_region['start'] <= pos <= elim_region['end']:
                keep_mask[i] = False
                break
    
    return positions[keep_mask], values[keep_mask]

def get_chromosome_labels_for_region(row_start, row_end, germ_soma_mapping, as_pu_mapping):
    """Get Parascaris and Ascaris chromosome names for a genomic region."""
    # Get Parascaris chromosomes in this region
    pu_chrs = []
    as_chrs = []
    
    for _, mapping in germ_soma_mapping.iterrows():
        if mapping['germ_start'] < row_end and mapping['germ_end'] > row_start:
            pu_chr = mapping['soma_chr']
            pu_chrs.append(pu_chr)
            
            # Find corresponding Ascaris chromosome(s)
            as_mappings = as_pu_mapping[as_pu_mapping['parascaris_chr'] == pu_chr]
            for _, as_mapping in as_mappings.iterrows():
                as_chrs.append(as_mapping['ascaris_chr'])
    
    # Remove duplicates while preserving order
    pu_chrs = list(dict.fromkeys(pu_chrs))
    as_chrs = list(dict.fromkeys(as_chrs))
    
    return pu_chrs, as_chrs

def chromosome_ranges_to_genomic_ranges(chr_names, germ_soma_mapping, chromosome):
    """
    Convert chromosome names to genomic ranges.
    Supports multiple formats:
    - Single chr per row: "chr01;chr02;chr03"
    - Multiple chrs per row: "chr01,chr02,chr03;chr04,chr05,chr06"
    - Ranges: "chr01-chr08;chrX1-chrX9"
    """
    ranges = []
    
    # Split by semicolon to get rows
    for row_spec in chr_names.split(';'):
        row_spec = row_spec.strip()
        if not row_spec:
            continue
        
        # Check if this is a range (e.g., "chr01-chr08")
        if '-' in row_spec and ',' not in row_spec:
            # This is a range specification
            parts = row_spec.split('-')
            if len(parts) == 2:
                start_chr = parts[0].strip()
                end_chr = parts[1].strip()
                
                # Find all chromosomes between start and end
                start_found = False
                end_found = False
                row_chrs = []
                
                for _, mapping in germ_soma_mapping[germ_soma_mapping['germ_chr'] == chromosome].iterrows():
                    chr_name = mapping['soma_chr']
                    if chr_name == start_chr:
                        start_found = True
                    if start_found:
                        row_chrs.append(chr_name)
                    if chr_name == end_chr:
                        end_found = True
                        break
                
                if not (start_found and end_found):
                    print(f"Warning: Range {row_spec} not fully found in mappings")
                    continue
            else:
                row_chrs = [row_spec]
        else:
            # This is a comma-separated list of chromosomes for one row
            row_chrs = [c.strip() for c in row_spec.split(',')]
        
        # Find genomic coordinates for all chromosomes in this row
        row_starts = []
        row_ends = []
        
        for chr_name in row_chrs:
            mapping = germ_soma_mapping[
                (germ_soma_mapping['germ_chr'] == chromosome) &
                (germ_soma_mapping['soma_chr'] == chr_name)
            ]
            
            if not mapping.empty:
                row = mapping.iloc[0]
                row_starts.append(int(row['germ_start']))
                row_ends.append(int(row['germ_end']))
            else:
                print(f"Warning: Chromosome {chr_name} not found in mappings")
        
        # Create a single range spanning all chromosomes in this row
        if row_starts and row_ends:
            ranges.append((min(row_starts), max(row_ends)))
    
    return ranges

def calculate_correlation_stats(pu_full_data, as_full_data, pu_chr, as_chr,
                               orientation, pu_flip, as_flip):
    """
    Calculate Pearson correlation between full chromosome datasets.
    Uses FULL chromosome data, not just visible segments.
    Follows exact logic from plot_eigenvector_comparison.py.
    """
    from scipy import stats
    
    if pu_full_data is None or as_full_data is None:
        return None
    
    # Ensure we're working with DataFrames
    if not isinstance(pu_full_data, pd.DataFrame):
        return None
    if not isinstance(as_full_data, pd.DataFrame):
        return None
    
    # Get data for this specific chromosome pair
    pu_chr_data = pu_full_data[pu_full_data['chr'] == pu_chr].copy()
    as_chr_data = as_full_data[as_full_data['chr'] == as_chr].copy()
    
    if len(pu_chr_data) == 0 or len(as_chr_data) == 0:
        return None
    
    # Create position/eigenvector dataframes
    pu_data = pd.DataFrame({
        'position': (pu_chr_data['start'] + pu_chr_data['end']) / 2,
        'eigenvector': pu_chr_data['eigenvector'].values
    })
    
    as_data = pd.DataFrame({
        'position': (as_chr_data['start'] + as_chr_data['end']) / 2,
        'eigenvector': as_chr_data['eigenvector'].values
    })
    
    # Apply Y-axis flips
    if pu_flip:
        pu_data['eigenvector'] = -pu_data['eigenvector']
    if as_flip:
        as_data['eigenvector'] = -as_data['eigenvector']
    
    # Handle X-axis reversal for Ascaris (for correlation calculation)
    if orientation == 'reverse':
        as_max = as_data['position'].max()
        as_data['position'] = as_max - as_data['position']
        as_data = as_data.sort_values('position').reset_index(drop=True)
    
    # Find overlapping coordinate range
    min_pos = max(pu_data['position'].min(), as_data['position'].min())
    max_pos = min(pu_data['position'].max(), as_data['position'].max())
    
    if min_pos >= max_pos:
        return None
    
    # Create common positions for interpolation
    common_positions = np.linspace(min_pos, max_pos, 100)
    
    # Interpolate both datasets to common positions
    pu_interp = np.interp(common_positions, pu_data['position'], pu_data['eigenvector'])
    as_interp = np.interp(common_positions, as_data['position'], as_data['eigenvector'])
    
    # Remove NaN values
    valid_idx = ~(np.isnan(pu_interp) | np.isnan(as_interp))
    pu_clean = pu_interp[valid_idx]
    as_clean = as_interp[valid_idx]
    
    if len(pu_clean) < 2 or len(as_clean) < 2:
        return None
    
    # Calculate Pearson correlation
    try:
        pearson_corr = np.corrcoef(pu_clean, as_clean)[0, 1]
        return pearson_corr
    except (ValueError, FloatingPointError):
        return None

def plot_row_segment(ax_top, ax_bottom, top_positions, top_values, bottom_positions, bottom_values,
                     elim_regions, germ_mappings, as_pu_mapping, row_start, row_end, 
                     y_min, y_max, show_ylabel=True, linewidth=2.5, top_color='red', bottom_color='#0066CC',
                     parascaris_flip=None, ascaris_flip=None, pu_full_data=None, as_full_data=None):
    """Plot a single row segment with top species on top, bottom species on bottom."""
    
    # Plot elimination regions (no borders, just shading)
    for _, region in elim_regions.iterrows():
        ax_top.axvspan(region['start']/1e6, region['end']/1e6, 
                     color='red', alpha=0.2, linewidth=0, zorder=1)
        ax_bottom.axvspan(region['start']/1e6, region['end']/1e6, 
                     color='red', alpha=0.2, linewidth=0, zorder=1)
    
    # Plot top data (Ascaris)
    if len(top_positions) > 0:
        if len(top_positions) > 1:
            gaps = np.diff(top_positions)
            median_gap = np.median(gaps)
            break_points = np.where(gaps > median_gap * 2)[0]
            
            start_idx = 0
            for break_idx in break_points:
                end_idx = break_idx + 1
                ax_top.plot(top_positions[start_idx:end_idx]/1e6, 
                          top_values[start_idx:end_idx], 
                          color=top_color, linewidth=linewidth, alpha=0.8, zorder=3)
                start_idx = end_idx
            ax_top.plot(top_positions[start_idx:]/1e6, 
                      top_values[start_idx:], 
                      color=top_color, linewidth=linewidth, alpha=0.8, zorder=3)
        else:
            ax_top.plot(top_positions/1e6, top_values, 
                      color=top_color, linewidth=linewidth, alpha=0.8, zorder=3)
    
    # Plot bottom data (Parascaris)
    if len(bottom_positions) > 0:
        if len(bottom_positions) > 1:
            gaps = np.diff(bottom_positions)
            median_gap = np.median(gaps)
            break_points = np.where(gaps > median_gap * 2)[0]
            
            start_idx = 0
            for break_idx in break_points:
                end_idx = break_idx + 1
                ax_bottom.plot(bottom_positions[start_idx:end_idx]/1e6, 
                          bottom_values[start_idx:end_idx],
                          color=bottom_color, linewidth=linewidth, alpha=0.8, zorder=3)
                start_idx = end_idx
            ax_bottom.plot(bottom_positions[start_idx:]/1e6, 
                      bottom_values[start_idx:],
                      color=bottom_color, linewidth=linewidth, alpha=0.8, zorder=3)
        else:
            ax_bottom.plot(bottom_positions/1e6, bottom_values,
                      color=bottom_color, linewidth=linewidth, alpha=0.8, zorder=3)
    
    # Collect correlations for this row
    row_correlations = []
    
    # Add chromosome labels and statistics
    for _, mapping in germ_mappings.iterrows():
        if mapping['germ_start'] < row_end and mapping['germ_end'] > row_start:
            label_start = max(mapping['germ_start'], row_start)
            label_end = min(mapping['germ_end'], row_end)
            label_center = (label_start + label_end) / 2
            
            # Find Ascaris chromosome for top panel
            as_mappings = as_pu_mapping[as_pu_mapping['parascaris_chr'] == mapping['soma_chr']]
            if not as_mappings.empty:
                as_chr_name = as_mappings.iloc[0]['ascaris_chr']
                orientation = as_mappings.iloc[0]['orientation']
                
                # Add "(R)" if reversed
                as_display_name = as_chr_name
                if orientation == 'reverse':
                    as_display_name = f"{as_chr_name} (R)"
                
                # Get flip information for this chromosome
                pu_flip = False
                as_flip = False
                if parascaris_flip and mapping['soma_chr'] in parascaris_flip:
                    pu_flip = parascaris_flip[mapping['soma_chr']]
                if ascaris_flip:
                    for chr_format in [as_chr_name, as_chr_name.replace('chr', ''), 'chr' + as_chr_name]:
                        if chr_format in ascaris_flip:
                            as_flip = ascaris_flip[chr_format]
                            break
                
                # Calculate statistics for this chromosome pair using FULL data
                pearson = calculate_correlation_stats(
                    pu_full_data, as_full_data, mapping['soma_chr'], as_chr_name,
                    orientation, pu_flip, as_flip
                )
                
                # Display Pearson inside Parascaris plot at bottom center (not bold)
                if pearson is not None:
                    stats_text = f'r={pearson:.2f}'
                    # Position at bottom of Parascaris plot (y_min + small offset)
                    ax_bottom.text(label_center/1e6, y_min + 0.05 * (y_max - y_min), stats_text, 
                              ha='center', va='bottom', fontsize=8, 
                              color='black', zorder=10)
                    row_correlations.append(pearson)
                
                # Display Ascaris chromosome name above Ascaris plot (not bold)
                ax_top.text(label_center/1e6, 1.10, as_display_name, 
                          ha='center', va='center', fontsize=9, 
                          color=top_color, transform=ax_top.get_xaxis_transform(), zorder=10)
            
            # Display Parascaris chromosome name CLOSE to plot (not bold)
            ax_bottom.text(label_center/1e6, -0.15, mapping['soma_chr'], 
                      ha='center', va='center', fontsize=9, 
                      color=bottom_color, transform=ax_bottom.get_xaxis_transform(), zorder=10)
    
    # Add zero line
    ax_top.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)
    ax_bottom.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)
    
    # Format axes - NO xlim normalization, use actual data range
    ax_top.set_ylim(y_min, y_max)
    ax_bottom.set_ylim(y_min, y_max)
    ax_top.set_xlim(row_start/1e6, row_end/1e6)
    ax_bottom.set_xlim(row_start/1e6, row_end/1e6)
    
    ax_top.grid(True, alpha=0.3, linewidth=0.5, zorder=0)
    ax_bottom.grid(True, alpha=0.3, linewidth=0.5, zorder=0)
    ax_top.set_axisbelow(True)
    ax_bottom.set_axisbelow(True)
    
    # Remove bottom spine from top panel and top spine from bottom panel
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    
    # Add black horizontal line between Ascaris and Parascaris panels (thicker)
    ax_top.axhline(y=y_min, color='black', linewidth=2.0, zorder=10)
    
    # Remove x-ticks from top panel completely
    ax_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    # Y-axis labels
    ax_top.set_ylabel('EV1', fontsize=10, labelpad=8)
    ax_bottom.set_ylabel('EV1', fontsize=10, labelpad=8)
    
    # Format x-axis: add "Mb" to tick labels, pull down with extended tick marks
    ax_bottom.set_xlabel('')
    
    # Custom formatter to add "Mb" to tick labels
    from matplotlib.ticker import FuncFormatter
    def mb_formatter(x, pos):
        return f'{x:.0f} Mb'
    ax_bottom.xaxis.set_major_formatter(FuncFormatter(mb_formatter))
    
    # Position x-axis labels down with extended tick marks
    ax_bottom.tick_params(axis='x', pad=15, length=10, direction='out')
    
    return row_correlations

def parse_row_ranges(row_ranges_str):
    """Parse row ranges string. Format: "0-20,20-40,40-60" (in Mb)"""
    ranges = []
    for range_str in row_ranges_str.split(','):
        start, end = map(float, range_str.strip().split('-'))
        ranges.append((int(start * 1e6), int(end * 1e6)))
    return ranges

def plot_multirow_eigenvector(pu_ev_file, as_ev_file, elimination_bed, as_elimination_bed,
                               germ_soma_mapping_file, as_pu_mapping_file, flip_file,
                               output_prefix, chromosome='chrX', row_ranges="0-45,45.000001-100,100.000001-163,163.000001-240.1",
                               row_chromosomes=None, smooth=True, window_size=5, dpi=300, 
                               y_min=-3.0, y_max=3.0, linewidth=2.5, pu_color='#0066CC', as_color='#FF8C00'):
    """Create multi-row eigenvector plot with proportional row heights and optimized label placement."""
    
    import os
    output_dir = "ev1_multirow_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nLoading data files...")
    pu_soma_ev_data = read_eigenvector_file(pu_ev_file)
    as_ev_data = read_eigenvector_file(as_ev_file) if as_ev_file else None
    pu_elimination_regions = read_bed_file(elimination_bed)
    as_elimination_regions = read_bed_file(as_elimination_bed) if as_elimination_bed else pd.DataFrame()
    germ_soma_mapping = read_germ_soma_mapping(germ_soma_mapping_file) if germ_soma_mapping_file else pd.DataFrame()
    as_pu_mapping = read_chromosome_mapping(as_pu_mapping_file) if as_pu_mapping_file else pd.DataFrame()
    parascaris_flip, ascaris_flip = read_flip_file(flip_file) if flip_file else ({}, {})
    
    # Map data
    pu_germ_data = map_soma_to_germ_coordinates(pu_soma_ev_data, germ_soma_mapping, parascaris_flip, chromosome)
    as_germ_data = map_ascaris_to_prepde_coordinates(as_ev_data, as_pu_mapping, germ_soma_mapping, ascaris_flip, chromosome)
    
    if pu_germ_data is None or pu_germ_data.empty:
        print(f"Error: Could not map data to {chromosome}")
        return
    
    # Verify we have the original DataFrames for correlation calculation
    print(f"Parascaris full data type: {type(pu_soma_ev_data)}")
    print(f"Ascaris full data type: {type(as_ev_data)}")
    if isinstance(pu_soma_ev_data, pd.DataFrame):
        print(f"  Parascaris chromosomes: {pu_soma_ev_data['chr'].unique()[:5]}")
    if isinstance(as_ev_data, pd.DataFrame):
        print(f"  Ascaris chromosomes: {as_ev_data['chr'].unique()[:5]}")
    
    # Parse row ranges (either from genomic ranges or chromosome names)
    if row_chromosomes:
        print(f"\nConverting chromosome names to genomic ranges: {row_chromosomes}")
        row_ranges_list = chromosome_ranges_to_genomic_ranges(row_chromosomes, germ_soma_mapping, chromosome)
    else:
        row_ranges_list = parse_row_ranges(row_ranges)
    
    num_rows = len(row_ranges_list)
    
    # Extend rows to include eliminated regions after the last chromosome
    # BUT make sure we don't extend into the next row's chromosomes
    extended_row_ranges = []
    for i, (row_start, row_end) in enumerate(row_ranges_list):
        # Find the start of the next row to avoid overlap
        next_row_start = row_ranges_list[i + 1][0] if i + 1 < len(row_ranges_list) else float('inf')
        
        # Get the last chromosome in this row
        last_chr_end = row_end
        for _, mapping in germ_soma_mapping[germ_soma_mapping['germ_chr'] == chromosome].iterrows():
            if mapping['germ_start'] >= row_start and mapping['germ_end'] <= row_end:
                last_chr_end = max(last_chr_end, mapping['germ_end'])
        
        # Find eliminated regions after the last chromosome but before next row
        next_elim = pu_elimination_regions[
            (pu_elimination_regions['chr'] == chromosome) &
            (pu_elimination_regions['start'] >= last_chr_end) &
            (pu_elimination_regions['start'] < next_row_start)
        ]
        
        if not next_elim.empty:
            # Extend to include eliminated regions, but cap at next row start
            extended_end = min(next_elim['end'].max(), next_row_start - 1)
            extended_row_ranges.append((row_start, extended_end))
            print(f"  Extended row {i+1} from {row_end/1e6:.1f} to {extended_end/1e6:.1f} Mb")
        else:
            extended_row_ranges.append((row_start, row_end))
    
    row_ranges_list = extended_row_ranges
    
    print(f"\nCreating {num_rows}-row plot with proportional widths:")
    for i, (start, end) in enumerate(row_ranges_list, 1):
        span_mb = (end - start) / 1e6
        print(f"  Row {i}: {start/1e6:.1f} - {end/1e6:.1f} Mb (span: {span_mb:.1f} Mb)")
    
    # Create figure with manual subplot positioning for proportional widths
    fig_height = 3.5 * num_rows
    fig = plt.figure(figsize=(16, fig_height))
    
    # Calculate proportional widths based on genomic spans
    row_spans_mb = [(end - start) / 1e6 for start, end in row_ranges_list]
    max_span_mb = max(row_spans_mb)
    
    # Calculate vertical positions for each row
    # Each row gets equal vertical space with spacing between
    row_height = 0.85 / num_rows  # Total vertical space divided by rows
    spacing = 0.10 / max(1, num_rows - 1) if num_rows > 1 else 0  # Reduced spacing
    
    axes = []
    
    for row_idx in range(num_rows):
        # Calculate proportional width for this row
        span_mb = row_spans_mb[row_idx]
        width_proportion = span_mb / max_span_mb
        
        # Calculate positions (left, bottom, width, height)
        left = 0.08
        subplot_width = 0.85 * width_proportion  # Proportional to genomic span
        
        # Panel height (same for Ascaris and Parascaris)
        panel_height = (row_height - spacing) / 2
        
        # Bottom position for this row (counting from bottom)
        row_bottom = 0.08 + (num_rows - 1 - row_idx) * (row_height + spacing)
        
        # Ascaris panel (top of row)
        ax_as_bottom = row_bottom + panel_height
        ax_as = fig.add_axes([left, ax_as_bottom, subplot_width, panel_height])
        
        # Parascaris panel (bottom of row, directly below Ascaris)
        ax_pu = fig.add_axes([left, row_bottom, subplot_width, panel_height])
        
        axes.extend([ax_as, ax_pu])
    
    axes = np.array(axes).reshape(-1, 1)
    
    # Get germ mappings
    germ_mappings = germ_soma_mapping[germ_soma_mapping['germ_chr'] == chromosome]
    
    # Collect all correlations for averaging
    all_correlations = []
    
    # Plot each row
    for row_idx, (row_start, row_end) in enumerate(row_ranges_list):
        ax_as = axes[row_idx * 2, 0]      # Ascaris on top
        ax_pu = axes[row_idx * 2 + 1, 0]  # Parascaris on bottom
        
        # Filter data for this row
        pu_row_data = pu_germ_data[
            (pu_germ_data['position'] >= row_start) & 
            (pu_germ_data['position'] <= row_end)
        ].copy()
        
        elim_row = pu_elimination_regions[
            (pu_elimination_regions['chr'] == chromosome) &
            (pu_elimination_regions['end'] > row_start) &
            (pu_elimination_regions['start'] < row_end)
        ] if not pu_elimination_regions.empty else pd.DataFrame()
        
        # Prepare Parascaris data
        pu_positions = pu_row_data['position'].values
        pu_values = pu_row_data['eigenvector'].values
        
        should_flip_pu = parascaris_flip.get(chromosome, False)
        if should_flip_pu:
            pu_values = -pu_values
        
        if smooth and len(pu_values) > 0:
            pu_values = smooth_data(pd.Series(pu_values), window_size).values
        
        pu_masked_pos, pu_masked_val = mask_eliminated_regions(pu_positions, pu_values, elim_row)
        
        # Prepare Ascaris data
        as_masked_pos = np.array([])
        as_masked_val = np.array([])
        
        if as_germ_data is not None and not as_germ_data.empty:
            as_row_data = as_germ_data[
                (as_germ_data['position'] >= row_start) & 
                (as_germ_data['position'] <= row_end)
            ].copy()
            
            if not as_row_data.empty:
                as_positions = as_row_data['position'].values
                as_values = as_row_data['eigenvector'].values
                
                if smooth:
                    as_values = smooth_data(pd.Series(as_values), window_size).values
                
                as_masked_pos, as_masked_val = mask_eliminated_regions(as_positions, as_values, elim_row)
        
        # Plot this row (Ascaris on top, Parascaris on bottom)
        show_ylabel = (row_idx == 0)
        
        row_correlations = plot_row_segment(ax_as, ax_pu, as_masked_pos, as_masked_val,
                        pu_masked_pos, pu_masked_val, elim_row, germ_mappings,
                        as_pu_mapping, row_start, row_end, y_min, y_max, 
                        show_ylabel, linewidth, as_color, pu_color,
                        parascaris_flip, ascaris_flip, pu_soma_ev_data, as_ev_data)
        
        # Collect correlations
        all_correlations.extend(row_correlations)
    
    # Calculate average correlation
    avg_correlation = np.mean(all_correlations) if len(all_correlations) > 0 else 0
    
    # Add legend with italicized species names
    legend_elements = [
        plt.Line2D([0], [0], color=as_color, lw=2.5, label=r'$\it{Ascaris}$'),
        plt.Line2D([0], [0], color=pu_color, lw=2.5, label=r'$\it{Parascaris}$'),
        mpatches.Patch(color='red', alpha=0.2, label='Eliminated regions')
    ]
    
    axes[0, 0].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
                     loc='upper left', fontsize=9)
    
    # Save
    png_file = os.path.join(output_dir, f'{output_prefix}_{chromosome}_multirow.png')
    plt.savefig(png_file, dpi=dpi, bbox_inches='tight')
    print(f"\nSaved: {png_file}")
    
    svg_file = os.path.join(output_dir, f'{output_prefix}_{chromosome}_multirow.svg')
    plt.savefig(svg_file, format='svg', bbox_inches='tight')
    print(f"Saved: {svg_file}")
    
    plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Plot multi-row eigenvector comparison'
    )
    
    parser.add_argument('pu_eigenvector', help='Parascaris post-PDE eigenvector file')
    parser.add_argument('--as-eigenvector', help='Ascaris eigenvector file')
    parser.add_argument('--elimination-bed', default='pu_v2_eliminated.bed',
                       help='BED file with pre-PDE eliminated regions')
    parser.add_argument('--as-elimination-bed', help='Ascaris elimination BED file')
    parser.add_argument('--germ-soma-mapping', default='germ_to_soma_mapping.bed',
                       help='Pre-PDE to post-PDE mapping')
    parser.add_argument('--as-pu-mapping', default='as_to_pu_chrom_order.txt',
                       help='Ascaris to Parascaris chromosome mapping')
    parser.add_argument('--flip-file', default='flip_or_not.txt',
                       help='Y-axis flip information file')
    parser.add_argument('--chromosome', default='chrX', help='Chromosome to plot')
    parser.add_argument('--row-ranges', default='0-45,45.000001-100,100.000001-163,163.000001-240.1',
                       help='Comma-separated ranges in Mb (e.g., "0-20,20-40,40-60")')
    parser.add_argument('--row-chromosomes',
                       help='Define rows by chromosome names (e.g., "chrX1,chrX2,chrX3")')
    parser.add_argument('--output', default='multirow_comparison',
                       help='Output file prefix')
    parser.add_argument('--smooth', action='store_true', default=True)
    parser.add_argument('--no-smooth', dest='smooth', action='store_false')
    parser.add_argument('--window', type=int, default=5, help='Smoothing window size')
    parser.add_argument('--dpi', type=int, default=300, help='Output resolution')
    parser.add_argument('--y-min', type=float, default=-3.0, help='Min Y-axis value')
    parser.add_argument('--y-max', type=float, default=3.0, help='Max Y-axis value')
    parser.add_argument('--linewidth', type=float, default=2.5, help='Line width for traces')
    parser.add_argument('--pu-color', default='#0066CC', help='Color for Parascaris traces (hex or name)')
    parser.add_argument('--as-color', default='#FF8C00', help='Color for Ascaris traces (hex or name)')
    
    args = parser.parse_args()
    
    plot_multirow_eigenvector(
        args.pu_eigenvector,
        args.as_eigenvector,
        args.elimination_bed,
        args.as_elimination_bed,
        args.germ_soma_mapping,
        args.as_pu_mapping,
        args.flip_file,
        args.output,
        args.chromosome,
        args.row_ranges,
        args.row_chromosomes,
        args.smooth,
        args.window,
        args.dpi,
        args.y_min,
        args.y_max,
        args.linewidth,
        args.pu_color,
        args.as_color
    )
    
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()
