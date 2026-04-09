#!/usr/bin/env python3
"""
analyze_ends_timepoints.py
==========================
Analyze multi-omics signal at new chromosome ends (created by programmed DNA
elimination) vs. internal break regions across Ascaris developmental timepoints.

For each window size, the script loads a BED file defining end and middle
(internal) break regions, then compares signal levels between these two region
types across 5 developmental stages (1-cell through L1) for:
  - Insulation scores (from FAN-C)
  - RNA-seq (gene expression)
  - ATAC-seq (chromatin accessibility)
  - H3K9me3 (heterochromatin)
  - H3K4me3 (active chromatin)

Outputs (per window size):
  - Multi-panel figure with dot+errorbar plots (PNG + SVG)
  - Statistical summary with Mann-Whitney U tests (text file)

Dependencies:
  numpy, pandas, matplotlib, seaborn, scipy

Example usage:
  python analyze_ends_timepoints.py \\
      --ends-dir data/new_ends_post_pde \\
      --window-sizes '100kb,250kb,500kb,1000kb' \\
      --plot-style points \\
      --error-bar-type sem \\
      --ylim-insulation '-4,2' \\
      --ylim-rnaseq '0,500' \\
      --ylim-atac '0,30' \\
      --ylim-h3k9me3 '0,300' \\
      --ylim-h3k4me3 '0,400' \\
      --output-dir results/

Input files:
  --ends-dir: Directory containing BED files of new chromosome end and
      internal break regions at each window size.  Filename pattern:
      AG.v50.new_ends_post_pde.{window_size}.bed
      Format: chr  start  end  name  type (where type = 'end' or 'middle')
  Omics bedgraph files in subdirectories relative to the working directory:
      insulation_scores/{prepde,postpde_mapped}/*.bedGraph
      rnaseq/50kb_binning/*.bg
      atac/50kb_binning/*.bg
      h3k9me3/50kb_binning/*.bg
      h3k4me3/50kb_binning/*.bg
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.patches as mpatches
from glob import glob

# Set style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['svg.fonttype'] = 'none'  # editable text in SVG for Illustrator

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze chromosome ends across timepoints')
    parser.add_argument('--ends-dir', type=str, default='data/new_ends_post_pde',
                        help='Directory containing new-ends BED files')
    parser.add_argument('--window-sizes', type=str, default='100kb,250kb,500kb,1000kb',
                        help='Comma-separated list of window sizes (e.g., 10kb,20kb,50kb)')
    parser.add_argument('--output-dir', type=str, default='plots',
                        help='Output directory for plots and statistics')
    parser.add_argument('--plot-style', type=str, default='points', choices=['violin', 'points'],
                        help='Plot style: violin plots or points with error bars')
    parser.add_argument('--error-bar-type', type=str, default='sem', choices=['sem', 'std'],
                        help='Error bar type: sem (standard error) or std (standard deviation)')
    parser.add_argument('--ylim-insulation', type=str, default='auto',
                        help='Y-axis limits for insulation (format: min,max or auto)')
    parser.add_argument('--ylim-rnaseq', type=str, default='auto',
                        help='Y-axis limits for RNA-seq (format: min,max or auto)')
    parser.add_argument('--ylim-atac', type=str, default='auto',
                        help='Y-axis limits for ATAC-seq (format: min,max or auto)')
    parser.add_argument('--ylim-h3k9me3', type=str, default='auto',
                        help='Y-axis limits for H3K9me3 (format: min,max or auto)')
    parser.add_argument('--ylim-h3k4me3', type=str, default='auto',
                        help='Y-axis limits for H3K4me3 (format: min,max or auto)')
    return parser.parse_args()

def parse_ylim(ylim_str):
    """Parse y-limit string - handles negative numbers"""
    if ylim_str.lower() == 'auto':
        return None
    try:
        # Remove any quotes and whitespace
        ylim_str = ylim_str.strip().strip('"').strip("'")
        parts = ylim_str.split(',')
        if len(parts) == 2:
            ymin = float(parts[0].strip())
            ymax = float(parts[1].strip())
            return (ymin, ymax)
    except Exception as e:
        print(f"Warning: Could not parse ylim string '{ylim_str}': {e}")
        return None
    return None

def normalize_window_size(window_size_str):
    """Normalize window size string (convert 1Mb to 1000kb for consistent naming)"""
    ws = window_size_str.lower().strip()
    if ws == '1mb':
        return '1000kb'
    return window_size_str.strip()

def read_bedgraph(filepath):
    """Read bedGraph file"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('track') or line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                try:
                    data.append({
                        'chr': parts[0],
                        'start': int(parts[1]),
                        'end': int(parts[2]),
                        'score': float(parts[3]) if parts[3] != 'NA' else np.nan
                    })
                except (ValueError, IndexError):
                    continue
    
    return pd.DataFrame(data)

def read_new_ends(bed_file):
    """Read new ends bed file"""
    ends = []
    with open(bed_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    ends.append({
                        'chr': parts[0],
                        'start': int(parts[1]),
                        'end': int(parts[2]),
                        'name': parts[3],
                        'type': parts[4].strip()  # 'end' or 'middle'
                    })
                elif len(parts) >= 3:
                    # If only 3 columns, infer type from name
                    name = parts[3] if len(parts) > 3 else f"{parts[0]}_region"
                    region_type = 'end' if 'end' in name.lower() or len(parts) == 3 else 'middle'
                    ends.append({
                        'chr': parts[0],
                        'start': int(parts[1]),
                        'end': int(parts[2]),
                        'name': name,
                        'type': region_type
                    })
    return pd.DataFrame(ends)

def get_scores_for_regions(bedgraph_df, regions_df):
    """Get scores for specific regions"""
    scores_by_region = []
    
    for _, region in regions_df.iterrows():
        chr_name = region['chr']
        region_start = region['start']
        region_end = region['end']
        region_name = region['name']
        region_type = region['type']
        
        # Get all bins that overlap with this region
        chr_data = bedgraph_df[bedgraph_df['chr'] == chr_name]
        
        # Find overlapping bins
        overlapping = chr_data[
            (chr_data['start'] < region_end) & 
            (chr_data['end'] > region_start)
        ]
        
        if len(overlapping) > 0:
            # Calculate mean score for the region
            valid_scores = overlapping['score'].dropna()
            if len(valid_scores) > 0:
                scores_by_region.append({
                    'region': region_name,
                    'type': region_type,
                    'chr': chr_name,
                    'start': region_start,
                    'end': region_end,
                    'mean_score': valid_scores.mean(),
                    'median_score': valid_scores.median(),
                    'n_bins': len(valid_scores)
                })
    
    # Create dataframe with defined columns to handle empty case
    if len(scores_by_region) == 0:
        return pd.DataFrame(columns=['region', 'type', 'chr', 'start', 'end', 
                                     'mean_score', 'median_score', 'n_bins'])
    
    return pd.DataFrame(scores_by_region)

def create_timepoint_comparison_plot(all_data, dataset_name, dataset_label, window_size, ax, 
                                    plot_style='points', error_bar_type='sem', ylimits=None):
    """Create a plot comparing timepoints for a single dataset"""
    
    # Define colors for timepoints - green to yellow to orange to red gradient
    timepoint_colors = {
        '0hr': '#27AE60',      # Green
        '48hr': '#F39C12',     # Orange-yellow
        '60hr': '#E67E22',     # Orange
        '5day': '#E74C3C',     # Light red
        '10day': '#8B0000'     # Dark red
    }
    
    # Define display labels for cell stages
    timepoint_labels = {
        '0hr': '1 cell',
        '48hr': '2-4 cell',
        '60hr': '4-8 cell',
        '5day': '32-64 cell',
        '10day': 'L1'
    }
    
    # Prepare data for plotting
    plot_data = []
    timepoints = ['0hr', '48hr', '60hr', '5day', '10day']
    
    for tp in timepoints:
        if tp in all_data:
            df = all_data[tp]
            for region_type in ['end', 'middle']:
                type_data = df[df['type'] == region_type]['mean_score']
                for val in type_data:
                    plot_data.append({
                        'Score': val,
                        'Timepoint': tp,
                        'Type': region_type,
                        'Group': f'{region_type}_{tp}'
                    })
    
    if not plot_data:
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    # Apply y-limits if specified - mark outliers
    if ylimits is not None:
        ymin_orig, ymax_orig = ylimits
        # Handle inverted axes (e.g., insulation score with 4,-4)
        ymin = min(ymin_orig, ymax_orig)
        ymax = max(ymin_orig, ymax_orig)
        
        plot_df['original_score'] = plot_df['Score']
        plot_df['is_outlier'] = (plot_df['Score'] < ymin) | (plot_df['Score'] > ymax)
        plot_df['outlier_type'] = 'none'
        plot_df.loc[plot_df['Score'] < ymin, 'outlier_type'] = 'low'
        plot_df.loc[plot_df['Score'] > ymax, 'outlier_type'] = 'high'
        # Clip values for plotting
        plot_df['Score'] = plot_df['Score'].clip(ymin, ymax)
    else:
        plot_df['is_outlier'] = False
        plot_df['outlier_type'] = 'none'
        plot_df['original_score'] = plot_df['Score']
        ymin_orig = None
        ymax_orig = None
    
    if plot_style == 'points':
        # Plot with points and error bars
        positions = []
        x_pos = 0
        
        for region_type in ['end', 'middle']:
            for i, tp in enumerate(timepoints):
                type_data = plot_df[(plot_df['Type'] == region_type) & 
                                   (plot_df['Timepoint'] == tp)]
                
                if len(type_data) > 0:
                    pos = x_pos + i * 0.22  # Tighter spacing for 5 timepoints
                    positions.append(pos)
                    
                    # Separate outliers and normal points
                    normal_data = type_data[~type_data['is_outlier']]
                    outliers = type_data[type_data['is_outlier']]
                    
                    # Calculate statistics from ORIGINAL values (before clipping)
                    all_scores = type_data['original_score'].values
                    mean_val = np.mean(all_scores)
                    if error_bar_type == 'sem':
                        error = stats.sem(all_scores)
                    else:  # std
                        error = np.std(all_scores)
                    
                    # Clip mean and error for display if needed
                    if ylimits is not None:
                        ymin, ymax = min(ymin_orig, ymax_orig), max(ymin_orig, ymax_orig)
                        mean_val_clipped = np.clip(mean_val, ymin, ymax)
                        
                        # Calculate how much of the error bar can be shown
                        # Ensure error bars are always non-negative
                        if mean_val - error < ymin:
                            # Lower error bar would go below ymin
                            error_lower = max(0, mean_val_clipped - ymin)
                        else:
                            error_lower = error
                        
                        if mean_val + error > ymax:
                            # Upper error bar would go above ymax
                            error_upper = max(0, ymax - mean_val_clipped)
                        else:
                            error_upper = error
                        
                        # Use the clipped mean for plotting
                        mean_val = mean_val_clipped
                        # Format as 2D array for matplotlib: [[lower], [upper]]
                        error = np.array([[error_lower], [error_upper]])
                    
                    # Plot normal points with jitter
                    if len(normal_data) > 0:
                        jitter = np.random.normal(0, 0.03, len(normal_data))
                        ax.scatter(pos + jitter, normal_data['Score'], 
                                 alpha=0.4, s=30, color=timepoint_colors[tp], 
                                 edgecolors='none', zorder=1)
                    
                    # Plot outliers as empty circles at limits
                    if len(outliers) > 0:
                        jitter = np.random.normal(0, 0.03, len(outliers))
                        ax.scatter(pos + jitter, outliers['Score'], 
                                 alpha=0.7, s=30, facecolors='none',
                                 edgecolors=timepoint_colors[tp], linewidths=2,
                                 zorder=1)
                    
                    # Plot mean with error bar
                    ax.errorbar(pos, mean_val, yerr=error, 
                              fmt='o', color=timepoint_colors[tp], 
                              markersize=10, capsize=6, capthick=2.5, 
                              linewidth=2.5, zorder=2, 
                              markeredgecolor='white', markeredgewidth=2)
            
            x_pos += 1.5
    
    else:  # violin plots (original style)
        for i, region_type in enumerate(['end', 'middle']):
            for j, tp in enumerate(timepoints):
                data_subset = plot_df[(plot_df['Type'] == region_type) & 
                                     (plot_df['Timepoint'] == tp)]['Score'].values
                
                if len(data_subset) > 0:
                    pos = i * 1.5 + j * 0.22
                    
                    # Violin plot
                    parts = ax.violinplot([data_subset], positions=[pos], 
                                         widths=0.2, showmeans=True)
                    
                    # Color the violin
                    for pc in parts['bodies']:
                        pc.set_facecolor(timepoint_colors[tp])
                        pc.set_alpha(0.7)
    
    # Set y-axis limits if specified - these are the FINAL limits
    if ylimits is not None:
        ax.set_ylim(ymin_orig, ymax_orig)
        y_min_plot, y_max_plot = ymin_orig, ymax_orig
    else:
        # Auto limits - get current then expand for stats
        y_min_auto, y_max_auto = ax.get_ylim()
        y_range_auto = y_max_auto - y_min_auto
        y_min_plot = y_min_auto - 0.15 * y_range_auto
        y_max_plot = y_max_auto + 0.35 * y_range_auto
        ax.set_ylim(y_min_plot, y_max_plot)
    
    y_range_plot = y_max_plot - y_min_plot
    
    # Add sample size labels ABOVE the data
    if plot_style == 'points':
        x_pos = 0
        for region_type in ['end', 'middle']:
            for i, tp in enumerate(timepoints):
                type_data = plot_df[(plot_df['Type'] == region_type) & 
                                   (plot_df['Timepoint'] == tp)]
                if len(type_data) > 0:
                    pos = x_pos + i * 0.22
                    n_outliers = len(type_data[type_data['is_outlier']])
                    n_total = len(type_data)
                    label_text = f'n={n_total}'
                    if n_outliers > 0:
                        label_text += f'\n({n_outliers}*)'
                    # Position above the data area
                    ax.text(pos, y_max_plot - 0.02 * y_range_plot, 
                           label_text, 
                           ha='center', va='top', fontsize=8, rotation=0,
                           clip_on=False)
            x_pos += 1.5
    else:  # violin style
        for i, region_type in enumerate(['end', 'middle']):
            for j, tp in enumerate(timepoints):
                data_subset = plot_df[(plot_df['Type'] == region_type) & 
                                     (plot_df['Timepoint'] == tp)]['Score'].values
                if len(data_subset) > 0:
                    pos = i * 1.5 + j * 0.22
                    ax.text(pos, y_max_plot - 0.02 * y_range_plot, 
                           f'n={len(data_subset)}', 
                           ha='center', va='top', fontsize=8, rotation=0,
                           clip_on=False)
    
    # Add labels
    ax.set_xticks([0.44, 1.94])  # Centered for 5 timepoints (0.22*4/2 = 0.44)
    ax.set_xticklabels([])  # Remove tick labels
    ax.tick_params(axis='y', labelsize=11)
    
    # Add End/Middle labels BELOW the axis
    ax.text(0.44, y_min_plot - 0.12 * y_range_plot, 'End', 
           ha='center', va='top', fontsize=14, fontweight='bold', clip_on=False)
    ax.text(1.94, y_min_plot - 0.12 * y_range_plot, 'Middle', 
           ha='center', va='top', fontsize=14, fontweight='bold', clip_on=False)
    
    # Add secondary x-axis labels for timepoints (above End/Middle labels, below axis)
    for i, region_type in enumerate(['end', 'middle']):
        for j, tp in enumerate(timepoints):
            # Check if this combination has data
            type_data = plot_df[(plot_df['Type'] == region_type) & 
                               (plot_df['Timepoint'] == tp)]
            if len(type_data) > 0:
                ax.text(i * 1.5 + j * 0.22, y_min_plot - 0.03 * y_range_plot, 
                       timepoint_labels[tp], ha='center', va='top', fontsize=10, rotation=0, 
                       color=timepoint_colors[tp], fontweight='bold', clip_on=False)
    
    # Statistical comparisons within each region type
    # With 5 timepoints, focus on key comparisons
    stat_spacing = [0.06, 0.13, 0.20, 0.27, 0.34]  # 5 levels for 5 comparisons
    
    for i, region_type in enumerate(['end', 'middle']):
        type_data = plot_df[plot_df['Type'] == region_type]
        
        # Only proceed if this region type has data
        if len(type_data) == 0:
            continue
        
        # Get data for each timepoint (use ORIGINAL scores for statistics)
        data_0hr = type_data[type_data['Timepoint'] == '0hr']['original_score'].values
        data_48hr = type_data[type_data['Timepoint'] == '48hr']['original_score'].values
        data_60hr = type_data[type_data['Timepoint'] == '60hr']['original_score'].values
        data_5day = type_data[type_data['Timepoint'] == '5day']['original_score'].values
        data_10day = type_data[type_data['Timepoint'] == '10day']['original_score'].values
        
        # Key comparisons: consecutive stages plus 0hr vs 10day
        comparisons = [
            (data_0hr, data_48hr, '0hr vs 48hr', 0, 0.22, 0),
            (data_48hr, data_60hr, '48hr vs 60hr', 0.22, 0.44, 1),
            (data_60hr, data_5day, '60hr vs 5day', 0.44, 0.66, 2),
            (data_5day, data_10day, '5day vs 10day', 0.66, 0.88, 3),
            (data_0hr, data_10day, '0hr vs 10day', 0, 0.88, 4)
        ]
        
        for data1, data2, label, offset1, offset2, comp_idx in comparisons:
            # Only compare if BOTH timepoints have data
            if len(data1) > 0 and len(data2) > 0:
                stat, p_val = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                
                # Draw comparison line above the axis limit
                y_pos = y_max_plot + (0.05 + stat_spacing[comp_idx]) * y_range_plot
                x1 = i * 1.5 + offset1
                x2 = i * 1.5 + offset2
                
                line = ax.plot([x1, x1, x2, x2], 
                       [y_pos - 0.01 * y_range_plot, y_pos, y_pos, y_pos - 0.01 * y_range_plot], 
                       'k-', linewidth=1.5, clip_on=False)
                
                # Add significance
                if p_val < 0.001:
                    sig_text = '***'
                elif p_val < 0.01:
                    sig_text = '**'
                elif p_val < 0.05:
                    sig_text = '*'
                else:
                    sig_text = 'ns'
                
                ax.text((x1 + x2) / 2, y_pos + 0.01 * y_range_plot, sig_text, 
                       ha='center', va='bottom', fontsize=10, fontweight='bold',
                       clip_on=False)
    
    # Add horizontal line at y=0 if appropriate (within axis limits)
    if y_min_plot < 0 < y_max_plot:
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Labels
    ax.set_ylabel(f'{dataset_label}', fontsize=13, fontweight='bold')
    ax.set_title(f'{dataset_label}', fontsize=14, fontweight='bold', pad=110)
    
    # Add grid
    ax.grid(True, alpha=0.4, axis='y', linewidth=0.8)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def create_comprehensive_figure(all_results, window_size, output_dir='plots', 
                                plot_style='points', error_bar_type='sem', ylimits_dict=None):
    """Create comprehensive figure with all datasets"""
    
    # Adjust layout to accommodate 5 datasets
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    axes = axes.flatten()
    
    # Dataset information
    datasets = [
        ('insulation', 'Insulation Score\n(Higher = Less Insulation)'),
        ('rnaseq', 'RNA-seq\n(Gene Expression)'),
        ('atac', 'ATAC-seq\n(Chromatin Accessibility)'),
        ('h3k9me3', 'H3K9me3\n(Heterochromatin Mark)'),
        ('h3k4me3', 'H3K4me3\n(Active Chromatin Mark)')
    ]
    
    plot_idx = 0
    for dataset_name, dataset_label in datasets:
        if dataset_name in all_results and plot_idx < 6:
            # Get ylimits for this dataset
            ylim = ylimits_dict.get(dataset_name, None) if ylimits_dict else None
            
            create_timepoint_comparison_plot(
                all_results[dataset_name], 
                dataset_name, 
                dataset_label, 
                window_size, 
                axes[plot_idx],
                plot_style=plot_style,
                error_bar_type=error_bar_type,
                ylimits=ylim
            )
            plot_idx += 1
    
    # Hide unused axes
    for i in range(plot_idx, 6):
        axes[i].set_visible(False)
    
    # Use the bottom right empty subplot for legend
    legend_ax = axes[5]
    legend_ax.set_visible(True)
    legend_ax.axis('off')
    
    # Create legend elements
    legend_elements = [
        mpatches.Patch(color='#27AE60', label='1 cell (pre-PDE)', linewidth=2, edgecolor='black'),
        mpatches.Patch(color='#F39C12', label='2-4 cell (pre-PDE)', linewidth=2, edgecolor='black'),
        mpatches.Patch(color='#E67E22', label='4-8 cell (pre-PDE)', linewidth=2, edgecolor='black'),
        mpatches.Patch(color='#E74C3C', label='32-64 cell (post-PDE)', linewidth=2, edgecolor='black'),
        mpatches.Patch(color='#8B0000', label='L1 (post-PDE)', linewidth=2, edgecolor='black')
    ]
    
    # Add legend to the empty subplot
    legend = legend_ax.legend(handles=legend_elements, loc='center', 
                             fontsize=16, frameon=True, 
                             fancybox=True, shadow=True,
                             title='Cell Stages', title_fontsize=18,
                             edgecolor='black', facecolor='white')
    legend.get_frame().set_linewidth(2)
    
    # Overall title
    error_type = 'SEM' if error_bar_type == 'sem' else 'SD'
    title_text = f'Multi-Omics Analysis at New Chromosome Ends ({window_size} windows)\n' + \
                 f'Comparing Cell Stages: 1 cell (pre-PDE) → 2-4 cell → 4-8 cell → 32-64 cell (post-PDE) → L1 (post-PDE)\n' + \
                 f'Error bars: mean ± {error_type}'
    
    # Add note about outliers if any ylimits are set
    if ylimits_dict and any(ylimits_dict.values()):
        title_text += ' | Empty circles indicate out-of-range values'
    
    fig.suptitle(title_text, fontsize=18, fontweight='bold')
    
    # Use tight_layout with more padding to accommodate labels outside axes
    plt.tight_layout(rect=[0, 0.02, 1, 0.96], pad=3.0, h_pad=4.0, w_pad=3.0)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'chromosome_ends_timepoints_{window_size}.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'chromosome_ends_timepoints_{window_size}.svg'), 
                format='svg', bbox_inches='tight')
    print(f"  Saved: {output_dir}/chromosome_ends_timepoints_{window_size}.png/svg")
    plt.close()

def save_statistics_summary(all_results, output_dir='plots'):
    """Save statistical summary of all comparisons"""
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'chromosome_ends_timepoint_stats.txt')
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Statistical Analysis of Chromosome Ends Across Cell Stages\n")
        f.write("="*80 + "\n\n")
        
        f.write("Comparisons: consecutive stages plus 0hr vs 10day\n")
        f.write("Test: Mann-Whitney U test (two-sided)\n")
        f.write("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant\n\n")
        
        datasets = ['insulation', 'rnaseq', 'atac', 'h3k9me3', 'h3k4me3']
        
        for window_size in sorted(all_results.keys()):
            f.write(f"\n{'='*80}\n")
            f.write(f"{window_size.upper()} WINDOW ANALYSIS\n")
            f.write("="*80 + "\n")
            
            for dataset in datasets:
                if dataset not in all_results[window_size]:
                    continue
                    
                f.write(f"\n{dataset.upper()}:\n")
                f.write("-"*60 + "\n")
                
                data = all_results[window_size][dataset]
                
                for region_type in ['end', 'middle']:
                    f.write(f"\n  {region_type.upper()} regions:\n")
                    
                    # Get data for each timepoint
                    timepoint_data = {}
                    for tp in ['0hr', '48hr', '60hr', '5day', '10day']:
                        if tp in data:
                            tp_values = data[tp][data[tp]['type'] == region_type]['mean_score'].values
                            timepoint_data[tp] = tp_values
                    
                    # Print summary stats
                    for tp, values in timepoint_data.items():
                        if len(values) > 0:
                            f.write(f"    {tp}: n={len(values)}, ")
                            f.write(f"mean={np.mean(values):.4f} ± {np.std(values):.4f}, ")
                            f.write(f"median={np.median(values):.4f}, ")
                            f.write(f"sem={stats.sem(values):.4f}\n")
                    
                    # Statistical comparisons
                    f.write("\n    Statistical comparisons:\n")
                    
                    comparisons = [
                        ('0hr', '48hr'),
                        ('48hr', '60hr'),
                        ('60hr', '5day'),
                        ('5day', '10day'),
                        ('0hr', '10day')
                    ]
                    
                    for tp1, tp2 in comparisons:
                        if tp1 in timepoint_data and tp2 in timepoint_data:
                            if len(timepoint_data[tp1]) > 0 and len(timepoint_data[tp2]) > 0:
                                stat, p_val = stats.mannwhitneyu(
                                    timepoint_data[tp1], 
                                    timepoint_data[tp2], 
                                    alternative='two-sided'
                                )
                                mean_diff = np.mean(timepoint_data[tp2]) - np.mean(timepoint_data[tp1])
                                f.write(f"      {tp1} vs {tp2}: p={p_val:.3e}, ")
                                f.write(f"mean_diff={mean_diff:+.4f}, U={stat:.1f}\n")
    
    print(f"  Saved: {output_file}")

def main():
    args = parse_arguments()
    
    # Parse window sizes - normalize all to kb format
    window_sizes_raw = [ws.strip() for ws in args.window_sizes.split(',')]
    window_sizes = [normalize_window_size(ws) for ws in window_sizes_raw]
    
    # Parse y-limits
    ylimits_dict = {
        'insulation': parse_ylim(args.ylim_insulation),
        'rnaseq': parse_ylim(args.ylim_rnaseq),
        'atac': parse_ylim(args.ylim_atac),
        'h3k9me3': parse_ylim(args.ylim_h3k9me3),
        'h3k4me3': parse_ylim(args.ylim_h3k4me3)
    }
    
    print("="*80)
    print("CHROMOSOME ENDS TIMEPOINT ANALYSIS")
    print("="*80)
    print(f"Window sizes: {', '.join(window_sizes)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Plot style: {args.plot_style}")
    print(f"Error bar type: {args.error_bar_type}")
    print(f"Y-axis limits:")
    for dataset, ylim in ylimits_dict.items():
        if ylim:
            print(f"  {dataset}: {ylim[0]} to {ylim[1]}")
        else:
            print(f"  {dataset}: auto")
    print("="*80)
    
    # Store all results
    all_results = {}
    
    for window_size in window_sizes:
        print(f"\n{'='*80}")
        print(f"PROCESSING {window_size.upper()} WINDOWS")
        print('='*80)
        
        # Try multiple possible filenames for the ends file
        possible_filenames = [
            os.path.join(args.ends_dir, f'AG.v50.new_ends_post_pde.{window_size}.bed'),
            os.path.join(args.ends_dir, f'AG.v50.new_ends_post_pde.1Mb.bed') if window_size == '1000kb' else None
        ]
        
        ends_file = None
        for filename in possible_filenames:
            if filename and os.path.exists(filename):
                ends_file = filename
                break
        
        if not ends_file:
            print(f"WARNING: File not found for {window_size}")
            print(f"  Tried: {[f for f in possible_filenames if f]}")
            continue
            
        new_ends = read_new_ends(ends_file)
        
        if len(new_ends) == 0:
            print(f"ERROR: No regions loaded from {ends_file}")
            continue
        
        # Debug: check what columns we have
        print(f"Columns in new_ends: {list(new_ends.columns)}")
        print(f"First few rows:")
        print(new_ends.head())
        
        print(f"Loaded {len(new_ends)} regions ({len(new_ends[new_ends['type']=='end'])} end, " +
              f"{len(new_ends[new_ends['type']=='middle'])} middle)")
        
        all_results[window_size] = {}
        
        # 1. Insulation scores
        print("\n1. Processing Insulation Scores...")
        insulation_files = {
            '0hr': 'insulation_scores/prepde/AG.v50.0hr.is100001.ids80001.insulation.bedGraph',
            '48hr': 'insulation_scores/prepde/AG.v50.48hr.is100001.ids80001.insulation.bedGraph',
            '60hr': 'insulation_scores/prepde/AG.v50.60hr.is100001.ids80001.insulation.bedGraph',
            '5day': 'insulation_scores/postpde_mapped/AG.v50_post.5day.is100001.ids80001.insulation.mapped.bedGraph',
            '10day': 'insulation_scores/postpde_mapped/AG.v50_post.10day.is100001.ids80001.insulation.mapped.bedGraph'
        }
        
        insulation_data = {}
        for timepoint, filepath in insulation_files.items():
            if os.path.exists(filepath):
                bg_data = read_bedgraph(filepath)
                insulation_data[timepoint] = get_scores_for_regions(bg_data, new_ends)
                print(f"  {timepoint}: {len(insulation_data[timepoint])} regions with data")
        
        if insulation_data:
            all_results[window_size]['insulation'] = insulation_data
        
        # 2. RNA-seq
        print("\n2. Processing RNA-seq...")
        rnaseq_files = {
            '0hr': 'rnaseq/50kb_binning/0hr.norm_50kb.bg',
            '48hr': 'rnaseq/50kb_binning/48hr.norm_50kb.bg',
            '60hr': 'rnaseq/50kb_binning/60hr.norm_50kb.bg',
            '5day': 'rnaseq/50kb_binning/5day.norm_50kb.bg',
            '10day': 'rnaseq/50kb_binning/10day.norm_50kb.bg'
        }
        
        rnaseq_data = {}
        for timepoint, filepath in rnaseq_files.items():
            if os.path.exists(filepath):
                bg_data = read_bedgraph(filepath)
                rnaseq_data[timepoint] = get_scores_for_regions(bg_data, new_ends)
                print(f"  {timepoint}: {len(rnaseq_data[timepoint])} regions with data")
        
        if rnaseq_data:
            all_results[window_size]['rnaseq'] = rnaseq_data
        
        # 3. ATAC-seq
        print("\n3. Processing ATAC-seq...")
        atac_data = {}
        
        for timepoint in ['0hr', '48hr', '60hr', '5day', '10day']:
            rep_files = glob(f'atac/50kb_binning/atac_{timepoint}_rep*.sorted_50kb.bg')
            
            if rep_files:
                rep_dfs = []
                for rep_file in rep_files:
                    if os.path.exists(rep_file):
                        rep_dfs.append(read_bedgraph(rep_file))
                
                if rep_dfs:
                    merged = rep_dfs[0].copy()
                    if len(rep_dfs) > 1:
                        for i in range(1, len(rep_dfs)):
                            merged = merged.merge(rep_dfs[i], 
                                                on=['chr', 'start', 'end'], 
                                                suffixes=('', f'_rep{i+1}'))
                        
                        score_cols = [col for col in merged.columns if 'score' in col]
                        merged['score'] = merged[score_cols].mean(axis=1)
                    
                    atac_data[timepoint] = get_scores_for_regions(merged, new_ends)
                    print(f"  {timepoint}: {len(atac_data[timepoint])} regions with data " +
                          f"(averaged {len(rep_files)} replicates)")
        
        if atac_data:
            all_results[window_size]['atac'] = atac_data
        
        # 4. H3K9me3
        print("\n4. Processing H3K9me3...")
        h3k9me3_data = {}
        
        h3k9me3_timepoint_map = {
            '0hr': 'H3K9me3_0hr_rep*.bg',
            '48hr': 'H3K9me3_48hr_rep*.bg',
            '60hr': 'H3K9me3_60hr_rep*.bg',
            '5day': 'H3K9me3_5day_rep*.bg',
            '10day': 'H3K9me3_10day_rep*.bg'
        }
        
        for timepoint, pattern in h3k9me3_timepoint_map.items():
            rep_files = glob(f'h3k9me3/50kb_binning/{pattern}')
            
            if rep_files:
                rep_dfs = []
                for rep_file in rep_files:
                    if os.path.exists(rep_file):
                        rep_dfs.append(read_bedgraph(rep_file))
                
                if rep_dfs:
                    merged = rep_dfs[0].copy()
                    if len(rep_dfs) > 1:
                        for i in range(1, len(rep_dfs)):
                            merged = merged.merge(rep_dfs[i], 
                                                on=['chr', 'start', 'end'], 
                                                suffixes=('', f'_rep{i+1}'))
                        
                        score_cols = [col for col in merged.columns if 'score' in col]
                        merged['score'] = merged[score_cols].mean(axis=1)
                    
                    h3k9me3_data[timepoint] = get_scores_for_regions(merged, new_ends)
                    print(f"  {timepoint}: {len(h3k9me3_data[timepoint])} regions with data " +
                          f"(averaged {len(rep_files)} replicates)")
            else:
                print(f"  {timepoint}: No data available")
        
        if h3k9me3_data:
            all_results[window_size]['h3k9me3'] = h3k9me3_data
        
        # 5. H3K4me3
        print("\n5. Processing H3K4me3...")
        h3k4me3_data = {}
        
        h3k4me3_timepoint_map = {
            '0hr': 'H3K4me3_0hr_rep*.bg',
            '48hr': 'H3K4me3_48hr_rep*.bg',
            '60hr': 'H3K4me3_60hr_rep*.bg',
            '5day': 'H3K4me3_5day_rep*_50kb.bg',
            '10day': 'H3K4me3_10day_rep*.bg'
        }
        
        for timepoint, pattern in h3k4me3_timepoint_map.items():
            rep_files = glob(f'h3k4me3/50kb_binning/{pattern}')
            
            if rep_files:
                rep_dfs = []
                for rep_file in rep_files:
                    if os.path.exists(rep_file):
                        rep_dfs.append(read_bedgraph(rep_file))
                
                if rep_dfs:
                    merged = rep_dfs[0].copy()
                    if len(rep_dfs) > 1:
                        for i in range(1, len(rep_dfs)):
                            merged = merged.merge(rep_dfs[i], 
                                                on=['chr', 'start', 'end'], 
                                                suffixes=('', f'_rep{i+1}'))
                        
                        score_cols = [col for col in merged.columns if 'score' in col]
                        merged['score'] = merged[score_cols].mean(axis=1)
                    
                    h3k4me3_data[timepoint] = get_scores_for_regions(merged, new_ends)
                    print(f"  {timepoint}: {len(h3k4me3_data[timepoint])} regions with data " +
                          f"(averaged {len(rep_files)} replicates)")
            else:
                print(f"  {timepoint}: No data available")
        
        if h3k4me3_data:
            all_results[window_size]['h3k4me3'] = h3k4me3_data
        
        # Create figure for this window size
        print(f"\nGenerating plots for {window_size}...")
        create_comprehensive_figure(all_results[window_size], window_size, 
                                   args.output_dir, args.plot_style, args.error_bar_type,
                                   ylimits_dict)
    
    # Save statistics summary
    print(f"\nSaving statistical summary...")
    save_statistics_summary(all_results, args.output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Output files saved to: {args.output_dir}/")
    print(f"  - chromosome_ends_timepoints_*.png/svg")
    print(f"  - chromosome_ends_timepoint_stats.txt")
    print("="*80)

if __name__ == '__main__':
    main()
