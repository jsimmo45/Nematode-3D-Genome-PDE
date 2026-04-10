#!/usr/bin/env python3
"""
plot_insulation.py
==================
Plot insulation scores across developmental stages for each Ascaris chromosome.
Each stage is stacked vertically with positive scores (strong boundaries) in
blue and negative scores (weak boundaries) in red.  Eliminated DNA regions are
shaded in red.  Post-PDE stages have data masked in eliminated regions.

Also produces a comprehensive box plot comparing insulation scores between
end, internal, and retained regions across all stages, and a statistical
summary with Mann-Whitney U tests.

Floor value removal: artificially low insulation scores (artifacts from
FAN-C at low-mappability bins) are detected and removed per region type
before analysis.

Outputs:
  - Per-chromosome stacked insulation plots (PNG, PDF, SVG)
  - Comprehensive box plot by region type (PNG, PDF, SVG)
  - Statistical summary CSV

Dependencies:
  numpy, pandas, matplotlib, seaborn, scipy

Example usage:
  python plot_insulation.py \\
      --prepde-dir insulation_scores/prepde \\
      --postpde-dir insulation_scores/postpde_mapped \\
      --eliminated-bed data/AG_v50_eliminated_strict.bed \\
      --prepde-stages teste ovary 0hr 48hr 60hr \\
      --postpde-stages 5day 10day \\
      --y-min -2 --y-max 1 \\
      --output-dir results/

Input files:
  --prepde-dir: Directory with pre-PDE insulation bedGraph files from FAN-C.
  --postpde-dir: Directory with post-PDE insulation bedGraph files mapped
      back to germline coordinates.
  --eliminated-bed: BED file of eliminated DNA regions (chr start end type).
      The type column should be 'end' or 'internal'.
"""
import os
import sys
import argparse
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set Arial as default font
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['svg.fonttype'] = 'none'  # editable text in SVG

# Set style
plt.style.use('default')
sns.set_palette("viridis")

def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced insulation score analysis")
    parser.add_argument('--prepde-dir', default='insulation_scores/prepde', 
                        help='Directory with pre-PDE insulation files')
    parser.add_argument('--postpde-dir', default='insulation_scores/postpde_mapped',
                        help='Directory with post-PDE mapped files') 
    parser.add_argument('--eliminated-bed', default='AG.v50.eliminated_strict.bed',
                        help='BED file with eliminated regions')
    parser.add_argument('--output-dir', default='insulation_analysis',
                        help='Output directory')
    parser.add_argument('--prepde-stages', nargs='*', 
                        default=['teste', 'ovary', 'germinal', 'middle', 'oocytes', '0hr', '48hr', '60hr'],
                        help='Pre-PDE stages to analyze in order')
    parser.add_argument('--postpde-stages', nargs='*', 
                        default=['4day', '5day', '10day'],
                        help='Post-PDE stages to analyze in order')
    parser.add_argument('--chromosomes', nargs='*', default=[], 
                        help='Chromosomes to analyze (leave empty for auto-detect all)')
    parser.add_argument('--y-min', type=float, default=-4,
                        help='Minimum y-axis value for chromosome insulation plots')
    parser.add_argument('--y-max', type=float, default=1,
                        help='Maximum y-axis value for chromosome insulation plots')
    parser.add_argument('--boxplot-y-min', type=float, default=-10,
                        help='Minimum y-axis value for boxplot (outliers shown as hollow)')
    parser.add_argument('--boxplot-y-max', type=float, default=2,
                        help='Maximum y-axis value for boxplot')
    parser.add_argument('--floor-threshold', type=float, default=-10,
                        help='Values below this are candidates for floor value removal')
    parser.add_argument('--floor-frequency', type=float, default=0.01,
                        help='Frequency threshold (0.01 = 1%) for floor value detection')
    return parser.parse_args()

def load_eliminated_regions(bed_file):
    """Load eliminated regions from BED file"""
    try:
        df = pd.read_csv(bed_file, sep='\t', header=None,
                        names=['chr', 'start', 'end', 'type'])
        df['start'] = df['start'].astype(int)
        df['end'] = df['end'].astype(int)
        
        print(f"\nLoaded eliminated regions: {len(df)} regions")
        type_counts = df['type'].value_counts()
        for elim_type, count in type_counts.items():
            print(f"  {elim_type}: {count} regions")
        
        # Convert to dictionary
        eliminated = {}
        for _, row in df.iterrows():
            chr_name = row['chr']
            if chr_name not in eliminated:
                eliminated[chr_name] = []
            eliminated[chr_name].append({
                'start': int(row['start']),
                'end': int(row['end']),
                'type': row['type']
            })
        
        # Sort regions
        for chr_name in eliminated:
            eliminated[chr_name].sort(key=lambda x: x['start'])
        
        return eliminated
    except Exception as e:
        print(f"Warning: Could not load eliminated regions from {bed_file}: {e}")
        return {}

def categorize_point(start, end, eliminated_regions, chr_name):
    """Categorize a single point as end, internal, or retained"""
    if chr_name not in eliminated_regions:
        return 'retained'
    
    bin_center = (start + end) / 2
    
    for elim_region in eliminated_regions[chr_name]:
        if elim_region['start'] <= bin_center <= elim_region['end']:
            return elim_region['type']
    
    return 'retained'

def load_and_clean_insulation_data(file_path, eliminated_regions, 
                                  floor_threshold=-10, floor_frequency=0.01):
    """
    Load insulation data and remove floor values PER CATEGORY in one pass.
    This is the ONLY place where floor values are removed.
    """
    print(f"\n  Loading {os.path.basename(file_path)}...")
    
    try:
        # Load the file
        if file_path.endswith('.bedGraph'):
            with open(file_path, 'r') as f:
                first_line = f.readline()
                if first_line.startswith('track'):
                    df = pd.read_csv(file_path, sep='\t', skiprows=1, header=None,
                                   names=['chr', 'start', 'end', 'insulation_score'])
                else:
                    df = pd.read_csv(file_path, sep='\t', header=None,
                                   names=['chr', 'start', 'end', 'insulation_score'])
        else:
            df = pd.read_csv(file_path, sep='\t')
            
            if 'insulation_score' not in df.columns:
                possible_cols = [col for col in df.columns 
                               if 'score' in col.lower() or 'insulation' in col.lower()]
                if possible_cols:
                    df = df.rename(columns={possible_cols[0]: 'insulation_score'})
                else:
                    df = pd.read_csv(file_path, sep='\t', header=None)
                    if len(df.columns) >= 4:
                        df = df.iloc[:, :4]
                        df.columns = ['chr', 'start', 'end', 'insulation_score']
        
        # Clean basic issues
        df['insulation_score'] = pd.to_numeric(df['insulation_score'], errors='coerce')
        df = df[~df['insulation_score'].isin([np.inf, -np.inf])]
        df = df.dropna(subset=['insulation_score'])
        df['start'] = df['start'].astype(int)
        df['end'] = df['end'].astype(int)
        
        print(f"    Loaded {len(df)} total points")
        
        # Add region type categorization
        df['region_type'] = df.apply(
            lambda row: categorize_point(row['start'], row['end'], 
                                        eliminated_regions, row['chr']), 
            axis=1
        )
        
        # Count points per category BEFORE cleaning
        print(f"    Before cleaning:")
        for rtype in ['end', 'internal', 'retained']:
            count = (df['region_type'] == rtype).sum()
            if count > 0:
                scores = df[df['region_type'] == rtype]['insulation_score']
                print(f"      {rtype}: {count} points, range [{scores.min():.1f}, {scores.max():.1f}]")
        
        # Detect and remove floor values PER CATEGORY
        cleaned_dfs = []
        
        for region_type in ['end', 'internal', 'retained']:
            type_df = df[df['region_type'] == region_type].copy()
            
            if len(type_df) == 0:
                continue
            
            # Find floor values for this category
            scores = type_df['insulation_score'].values
            unique, counts = np.unique(scores, return_counts=True)
            
            floor_values = []
            total = len(scores)
            for val, count in zip(unique, counts):
                freq = count / total
                if val < floor_threshold and freq > floor_frequency:
                    floor_values.append(val)
            
            if floor_values:
                print(f"      Removing floor values from {region_type}: {sorted(floor_values)[:5]}...")
                mask = ~type_df['insulation_score'].isin(floor_values)
                n_removed = (~mask).sum()
                print(f"        Removed {n_removed}/{len(type_df)} points ({n_removed/len(type_df)*100:.1f}%)")
                type_df = type_df[mask]
            
            cleaned_dfs.append(type_df)
        
        # Combine cleaned data
        df_clean = pd.concat(cleaned_dfs, ignore_index=True) if cleaned_dfs else pd.DataFrame()
        
        # Count points per category AFTER cleaning
        print(f"    After cleaning:")
        for rtype in ['end', 'internal', 'retained']:
            count = (df_clean['region_type'] == rtype).sum()
            if count > 0:
                scores = df_clean[df_clean['region_type'] == rtype]['insulation_score']
                print(f"      {rtype}: {count} points, range [{scores.min():.1f}, {scores.max():.1f}]")
        
        print(f"    Final: {len(df_clean)} points retained")
        
        return df_clean
        
    except Exception as e:
        print(f"    Error loading {file_path}: {e}")
        return pd.DataFrame()

def plot_chromosome_insulation(chr_name, stage_order, stage_data_dict, 
                              eliminated_regions, output_dir, y_min=-4, y_max=1):
    """Plot insulation scores for one chromosome with cleaned data"""
    
    print(f"\n  Plotting chromosome {chr_name}...")
    
    # Filter stages that have data for this chromosome
    valid_stages = []
    for stage in stage_order:
        if stage in stage_data_dict:
            chr_data = stage_data_dict[stage][stage_data_dict[stage]['chr'] == chr_name]
            if len(chr_data) > 0:
                valid_stages.append(stage)
    
    if not valid_stages:
        print(f"    No data for {chr_name}")
        return
    
    n_stages = len(valid_stages)
    fig, axes = plt.subplots(n_stages, 1, figsize=(16, 2*n_stages), sharex=True)
    if n_stages == 1:
        axes = [axes]
    
    post_pde_stages = ['4day', '5day', '10day']
    
    # Get x-axis range
    x_min, x_max = float('inf'), float('-inf')
    for stage in valid_stages:
        chr_data = stage_data_dict[stage][stage_data_dict[stage]['chr'] == chr_name]
        if len(chr_data) > 0:
            x_min = min(x_min, chr_data['start'].min())
            x_max = max(x_max, chr_data['end'].max())
    
    # Include eliminated regions in range
    if chr_name in eliminated_regions:
        for elim in eliminated_regions[chr_name]:
            x_min = min(x_min, elim['start'])
            x_max = max(x_max, elim['end'])
    
    # Plot each stage
    for i, stage in enumerate(valid_stages):
        ax = axes[i]
        
        # Get data for this chromosome (already cleaned)
        chr_data = stage_data_dict[stage][stage_data_dict[stage]['chr'] == chr_name].copy()
        
        # For post-PDE, remove points in eliminated regions COMPLETELY
        is_postpde = stage in post_pde_stages
        if is_postpde:
            # Only keep retained regions
            chr_data = chr_data[chr_data['region_type'] == 'retained']
        
        if len(chr_data) == 0:
            ax.text(0.5, 0.5, f'No data for {stage} after filtering', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12, family='Arial')
            ax.set_ylabel(stage, fontweight='bold', fontsize=14, family='Arial')
            continue
        
        # Sort by position
        chr_data = chr_data.sort_values('start')
        
        # For post-PDE: plot segments separately to avoid lines across eliminated regions
        if is_postpde and chr_name in eliminated_regions:
            # Group continuous segments (not interrupted by eliminated regions)
            segments = []
            current_segment = []
            
            for _, row in chr_data.iterrows():
                if not current_segment:
                    current_segment.append(row)
                else:
                    # Check if there's a gap (eliminated region) between last point and this one
                    last_end = current_segment[-1]['end']
                    gap_exists = False
                    
                    for elim_region in eliminated_regions[chr_name]:
                        if elim_region['start'] < row['start'] and elim_region['end'] > last_end:
                            gap_exists = True
                            break
                    
                    if gap_exists:
                        # Start new segment
                        segments.append(pd.DataFrame(current_segment))
                        current_segment = [row]
                    else:
                        current_segment.append(row)
            
            if current_segment:
                segments.append(pd.DataFrame(current_segment))
            
            # Plot each segment separately
            for segment_df in segments:
                if len(segment_df) > 0:
                    ax.plot(segment_df['start'], segment_df['insulation_score'],
                           color='black', linewidth=1.0, alpha=0.8)
                    
                    x_vals = segment_df['start'].values
                    y_vals = segment_df['insulation_score'].values
                    
                    ax.fill_between(x_vals, 0, y_vals, where=(y_vals >= 0),
                                   color='blue', alpha=0.3, interpolate=True)
                    ax.fill_between(x_vals, 0, y_vals, where=(y_vals < 0),
                                   color='red', alpha=0.3, interpolate=True)
        else:
            # Pre-PDE: plot continuous line (includes eliminated regions)
            ax.plot(chr_data['start'], chr_data['insulation_score'],
                   color='black', linewidth=1.0, alpha=0.8)
            
            x_vals = chr_data['start'].values
            y_vals = chr_data['insulation_score'].values
            
            ax.fill_between(x_vals, 0, y_vals, where=(y_vals >= 0),
                           color='blue', alpha=0.3, interpolate=True)
            ax.fill_between(x_vals, 0, y_vals, where=(y_vals < 0),
                           color='red', alpha=0.3, interpolate=True)
        
        # Add zero line
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Add eliminated regions as RED shaded areas (ALL THE SAME COLOR)
        if chr_name in eliminated_regions:
            for j, elim in enumerate(eliminated_regions[chr_name]):
                ax.axvspan(elim['start'], elim['end'], alpha=0.2, color='red', zorder=1,
                          label='Eliminated Regions' if i == 0 and j == 0 else "")
        
        # Formatting
        ax.set_ylabel(stage, fontweight='bold', fontsize=14, rotation=0, ha='right', va='center', family='Arial')
        ax.grid(True, alpha=0.2, axis='y')
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(x_min, x_max)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=12)
        
        if i == n_stages - 1:
            ax.set_xlabel(f'{chr_name} Position (bp)', fontweight='bold', fontsize=14, family='Arial')
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        
        # Legend on first plot
        if i == 0 and chr_name in eliminated_regions:
            ax.legend(loc='upper right', fontsize=10, prop={'family': 'Arial'})
    
    plt.suptitle(f'{chr_name} - Insulation Scores (Floor Values Removed)', fontweight='bold', fontsize=16, family='Arial')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'insulation_{chr_name}')
    plt.savefig(f'{output_file}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_file}.pdf', bbox_inches='tight')
    plt.savefig(f'{output_file}.svg', bbox_inches='tight')
    plt.close()
    
    print(f"    Saved {chr_name} plot (PNG, PDF, SVG)")

def create_comprehensive_boxplot(stage_data_dict, stage_order, output_dir, 
                                plot_y_min=-10, plot_y_max=2):
    """Create box plot from already-cleaned data with y-axis limits and significance bars"""
    
    print("\n  Creating comprehensive box plot...")
    
    # Combine all data
    all_data = []
    for stage in stage_order:
        if stage in stage_data_dict:
            df = stage_data_dict[stage].copy()
            df['stage'] = stage
            all_data.append(df)
    
    if not all_data:
        print("    No data for box plot")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    
    # Define colors
    colors = {
        'end': '#FF6B6B',
        'internal': '#4ECDC4',
        'retained': '#45B7D1'
    }
    
    # Define post-PDE stages for background shading
    post_pde_stages = ['4day', '5day', '10day']
    
    # Prepare box plot data
    positions = []
    box_data = []
    box_colors = []
    box_types = []  # Track which type each box is
    stage_centers = {}
    stage_box_positions = {}  # Track positions for significance bars
    
    pos = 0
    for stage in stage_order:
        stage_df = combined_df[combined_df['stage'] == stage]
        if len(stage_df) == 0:
            continue
        
        stage_start = pos
        stage_positions = {}
        
        for rtype in ['end', 'internal', 'retained']:
            type_data = stage_df[stage_df['region_type'] == rtype]['insulation_score'].values
            if len(type_data) > 0:
                box_data.append(type_data)
                positions.append(pos)
                box_colors.append(colors[rtype])
                box_types.append(rtype)
                stage_positions[rtype] = pos
                pos += 1
        
        if pos > stage_start:
            stage_centers[stage] = (stage_start + pos - 1) / 2
            stage_box_positions[stage] = stage_positions
            pos += 0.5  # Gap between stages
    
    if not box_data:
        print("    No data to plot")
        return
    
    # Add light red background for post-PDE stages BEFORE plotting data
    # Find the overall span of post-PDE stages for continuous highlighting
    post_pde_positions = []
    for stage in stage_centers:
        if stage in post_pde_stages:
            stage_pos_list = list(stage_box_positions[stage].values())
            post_pde_positions.extend(stage_pos_list)
    
    if post_pde_positions:
        left_edge = min(post_pde_positions) - 0.5
        right_edge = max(post_pde_positions) + 0.5
        ax.axvspan(left_edge, right_edge, alpha=0.15, color='red', zorder=0)
    
    # Create box plots
    bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=2))
    
    # Color boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    
    # Add sample points with special formatting for outliers
    np.random.seed(42)
    for data, pos, color in zip(box_data, positions, box_colors):
        n_show = min(len(data), 50)
        if n_show > 0:
            show_data = np.random.choice(data, n_show, replace=False) if len(data) > 50 else data
            jitter = np.random.normal(0, 0.08, len(show_data))
            x_pos = np.full(len(show_data), pos) + jitter
            
            # Separate points above and below threshold
            normal_mask = show_data >= plot_y_min
            outlier_mask = show_data < plot_y_min
            
            # Plot normal points (filled)
            if normal_mask.any():
                ax.scatter(x_pos[normal_mask], show_data[normal_mask], 
                          alpha=0.7, s=15, color=color, edgecolor='none')
            
            # Plot outliers (hollow with bold outline) at the y_min line
            if outlier_mask.any():
                ax.scatter(x_pos[outlier_mask], np.full(outlier_mask.sum(), plot_y_min),
                          alpha=1.0, s=25, facecolors='none', edgecolors=color, 
                          linewidths=2, marker='o')
                # Add small indicators for values below threshold
                for x in x_pos[outlier_mask]:
                    ax.plot(x, plot_y_min - 0.2, marker='v', markersize=5, 
                           color=color, alpha=0.5)
    
    # Set y-axis limits
    ax.set_ylim(plot_y_min, plot_y_max)
    
    # Add significance bars with asterisks
    y_top = plot_y_max
    bar_height = y_top * 0.05  # Height of significance bars above data
    
    for stage in stage_centers:
        stage_df = combined_df[combined_df['stage'] == stage]
        stage_pos = stage_box_positions[stage]
        
        # Count samples and add at top
        n_end = (stage_df['region_type'] == 'end').sum()
        n_int = (stage_df['region_type'] == 'internal').sum()
        n_ret = (stage_df['region_type'] == 'retained').sum()
        
        x_pos = stage_centers[stage]
        ax.text(x_pos, y_top * 0.96, f"n={n_end}/{n_int}/{n_ret}",
               ha='center', fontsize=10, fontweight='bold', family='Arial')
        
        # Calculate p-values and draw significance bars
        comparisons = [
            ('end', 'retained'),
            ('internal', 'retained'),
            ('end', 'internal')
        ]
        
        bar_y_start = y_top * 0.85  # Starting height for bars
        bar_increment = y_top * 0.08  # Space between bars
        
        for i, (type1, type2) in enumerate(comparisons):
            if type1 not in stage_pos or type2 not in stage_pos:
                continue
                
            data1 = stage_df[stage_df['region_type'] == type1]['insulation_score'].values
            data2 = stage_df[stage_df['region_type'] == type2]['insulation_score'].values
            
            if len(data1) >= 5 and len(data2) >= 5:
                _, p_val = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                
                # Only draw bar if significant
                if p_val < 0.05:
                    # Determine significance level
                    if p_val < 0.001:
                        sig_text = '***'
                    elif p_val < 0.01:
                        sig_text = '**'
                    else:
                        sig_text = '*'
                    
                    # Draw significance bar
                    x1 = stage_pos[type1]
                    x2 = stage_pos[type2]
                    bar_y = bar_y_start - (i * bar_increment)
                    
                    # Horizontal line
                    ax.plot([x1, x2], [bar_y, bar_y], 'k-', linewidth=1.5)
                    # Vertical ticks
                    tick_size = bar_height * 0.3
                    ax.plot([x1, x1], [bar_y - tick_size, bar_y], 'k-', linewidth=1.5)
                    ax.plot([x2, x2], [bar_y - tick_size, bar_y], 'k-', linewidth=1.5)
                    # Significance asterisks
                    ax.text((x1 + x2) / 2, bar_y + bar_height * 0.2, sig_text,
                           ha='center', va='bottom', fontsize=16, fontweight='bold', family='Arial')
    
    # Labels and formatting with bigger fonts
    ax.set_ylabel('Insulation Score', fontweight='bold', fontsize=18, family='Arial')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add line at y_min to show truncation
    ax.axhline(y=plot_y_min, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.text(ax.get_xlim()[1] * 0.98, plot_y_min + 0.1, f'Values <{plot_y_min} shown as hollow circles',
           ha='right', fontsize=11, style='italic', alpha=0.7, family='Arial')
    
    # X-axis with bigger fonts
    ax.set_xticks(list(stage_centers.values()))
    ax.set_xticklabels(list(stage_centers.keys()), fontweight='bold', fontsize=14, family='Arial')
    ax.set_xlabel('Developmental Stage', fontweight='bold', fontsize=18, family='Arial')
    
    # Y-axis tick labels bigger
    ax.tick_params(axis='y', labelsize=14)
    
    # Legend at bottom right, just above the truncation note
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['end'], alpha=0.7, label='End Eliminated'),
        Patch(facecolor=colors['internal'], alpha=0.7, label='Internal Eliminated'),
        Patch(facecolor=colors['retained'], alpha=0.7, label='Retained')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=14, prop={'family': 'Arial'},
             bbox_to_anchor=(0.98, 0.02))
    
    # Info box with significance legend - positioned just above the main legend
    sig_legend_text = ('* p<0.05  ** p<0.01  *** p<0.001')
    ax.text(0.98, 0.12, sig_legend_text,
           transform=ax.transAxes, fontsize=12, va='bottom', ha='right', family='Arial',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'comprehensive_boxplot')
    plt.savefig(f'{output_file}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_file}.pdf', bbox_inches='tight')
    plt.savefig(f'{output_file}.svg', bbox_inches='tight')
    plt.close()
    
    print("    Saved comprehensive box plot (PNG, PDF, SVG)")

def save_statistical_summary(stage_data_dict, stage_order, output_dir):
    """Calculate and save statistical summary from cleaned data"""
    
    print("\n  Generating statistical summary...")
    
    results = []
    
    for stage in stage_order:
        if stage not in stage_data_dict:
            continue
        
        df = stage_data_dict[stage]
        
        # Get data by type
        end_data = df[df['region_type'] == 'end']['insulation_score'].values
        int_data = df[df['region_type'] == 'internal']['insulation_score'].values
        ret_data = df[df['region_type'] == 'retained']['insulation_score'].values
        
        result = {
            'stage': stage,
            'n_end': len(end_data),
            'n_internal': len(int_data),
            'n_retained': len(ret_data),
        }
        
        # Calculate statistics
        if len(end_data) > 0:
            result['mean_end'] = np.mean(end_data)
            result['median_end'] = np.median(end_data)
            result['std_end'] = np.std(end_data)
        
        if len(int_data) > 0:
            result['mean_internal'] = np.mean(int_data)
            result['median_internal'] = np.median(int_data)
            result['std_internal'] = np.std(int_data)
        
        if len(ret_data) > 0:
            result['mean_retained'] = np.mean(ret_data)
            result['median_retained'] = np.median(ret_data)
            result['std_retained'] = np.std(ret_data)
        
        # Statistical tests
        min_n = 5
        if len(end_data) >= min_n and len(ret_data) >= min_n:
            _, p = stats.mannwhitneyu(end_data, ret_data, alternative='two-sided')
            result['p_end_vs_retained'] = p
            result['sig_end_vs_retained'] = 'Yes' if p < 0.05 else 'No'
        
        if len(int_data) >= min_n and len(ret_data) >= min_n:
            _, p = stats.mannwhitneyu(int_data, ret_data, alternative='two-sided')
            result['p_internal_vs_retained'] = p
            result['sig_internal_vs_retained'] = 'Yes' if p < 0.05 else 'No'
        
        if len(end_data) >= min_n and len(int_data) >= min_n:
            _, p = stats.mannwhitneyu(end_data, int_data, alternative='two-sided')
            result['p_end_vs_internal'] = p
            result['sig_end_vs_internal'] = 'Yes' if p < 0.05 else 'No'
        
        # Kruskal-Wallis
        groups = [d for d in [end_data, int_data, ret_data] if len(d) >= min_n]
        if len(groups) >= 2:
            _, p = stats.kruskal(*groups)
            result['p_kruskal'] = p
            result['sig_kruskal'] = 'Yes' if p < 0.05 else 'No'
        
        results.append(result)
    
    # Save to CSV
    df_results = pd.DataFrame(results)
    output_file = os.path.join(output_dir, 'statistical_summary.csv')
    df_results.to_csv(output_file, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("STATISTICAL SUMMARY")
    print("="*60)
    
    for _, row in df_results.iterrows():
        print(f"\n{row['stage']}:")
        print(f"  N: End={row['n_end']}, Internal={row['n_internal']}, Retained={row['n_retained']}")
        
        if 'median_end' in row and pd.notna(row['median_end']):
            print(f"  Medians: End={row['median_end']:.2f}, ", end='')
            if 'median_internal' in row and pd.notna(row['median_internal']):
                print(f"Internal={row['median_internal']:.2f}, ", end='')
            if 'median_retained' in row and pd.notna(row['median_retained']):
                print(f"Retained={row['median_retained']:.2f}")
        
        # Report significant tests
        sig_tests = []
        if 'sig_end_vs_retained' in row and row['sig_end_vs_retained'] == 'Yes':
            sig_tests.append(f"End vs Retained (p={row['p_end_vs_retained']:.3e})")
        if 'sig_internal_vs_retained' in row and row['sig_internal_vs_retained'] == 'Yes':
            sig_tests.append(f"Internal vs Retained (p={row['p_internal_vs_retained']:.3e})")
        if 'sig_end_vs_internal' in row and row['sig_end_vs_internal'] == 'Yes':
            sig_tests.append(f"End vs Internal (p={row['p_end_vs_internal']:.3e})")
        
        if sig_tests:
            print(f"  SIGNIFICANT: {', '.join(sig_tests)}")
    
    print(f"\nResults saved to: {output_file}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("ENHANCED INSULATION SCORE ANALYSIS")
    print("="*60)
    
    # Load eliminated regions ONCE
    eliminated_regions = load_eliminated_regions(args.eliminated_bed)
    
    # Combine stage order
    stage_order = args.prepde_stages + args.postpde_stages
    
    # Find stage files
    print("\nSearching for stage files...")
    stage_files = {}
    
    for stage in args.prepde_stages:
        for ext in ['.bedGraph', '.insulation']:
            pattern = os.path.join(args.prepde_dir, f'*{stage}*{ext}')
            files = glob(pattern)
            if files:
                stage_files[stage] = files[0]
                print(f"  {stage}: {os.path.basename(files[0])}")
                break
    
    for stage in args.postpde_stages:
        for ext in ['.bedGraph', '.insulation']:
            pattern = os.path.join(args.postpde_dir, f'*{stage}*{ext}')
            files = glob(pattern)
            if files:
                stage_files[stage] = files[0]
                print(f"  {stage}: {os.path.basename(files[0])}")
                break
    
    # Load and clean ALL data ONCE
    print("\nLoading and cleaning all data...")
    stage_data_dict = {}
    
    for stage in stage_order:
        if stage not in stage_files:
            print(f"\n  Skipping {stage} - no file found")
            continue
        
        # Load and clean in one pass
        cleaned_df = load_and_clean_insulation_data(
            stage_files[stage], 
            eliminated_regions,
            floor_threshold=args.floor_threshold,
            floor_frequency=args.floor_frequency
        )
        
        if len(cleaned_df) > 0:
            stage_data_dict[stage] = cleaned_df
    
    # Determine chromosomes to analyze
    if args.chromosomes:
        chromosomes = args.chromosomes
    else:
        chromosomes = set()
        for df in stage_data_dict.values():
            chromosomes.update(df['chr'].unique())
        chromosomes = sorted([c for c in chromosomes 
                             if not any(x in c for x in ['chrM', 'chrUn', 'random', 'alt', 'scaffold'])])
    
    print(f"\nChromosomes to analyze: {chromosomes}")
    
    # Generate chromosome plots
    print("\nGenerating chromosome plots...")
    for chr_name in chromosomes:
        plot_chromosome_insulation(chr_name, stage_order, stage_data_dict,
                                  eliminated_regions, args.output_dir,
                                  args.y_min, args.y_max)
    
    # Generate comprehensive box plot
    print("\nGenerating comprehensive box plot...")
    create_comprehensive_boxplot(stage_data_dict, stage_order, args.output_dir,
                                args.boxplot_y_min, args.boxplot_y_max)
    
    # Generate statistical summary
    print("\nGenerating statistical summary...")
    save_statistical_summary(stage_data_dict, stage_order, args.output_dir)
    
    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to: {args.output_dir}/")
    print("  - insulation_chr*.png/pdf/svg: Individual chromosome plots")
    print("  - comprehensive_boxplot.png/pdf/svg: Combined box plot")
    print("  - statistical_summary.csv: Statistical test results")
    
    # Count outputs
    n_chr_plots = len(glob(os.path.join(args.output_dir, 'insulation_chr*.png')))
    print(f"\nGenerated {n_chr_plots} chromosome plots (each in PNG, PDF, and SVG formats)")

if __name__ == '__main__':
    main()
