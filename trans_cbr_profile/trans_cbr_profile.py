#!/usr/bin/env python3
"""
trans_cbr_profile.py
====================
Analyze trans (inter-chromosomal) Hi-C interaction profiles around chromosome
break regions (CBRs) in Ascaris.  Counts trans contacts in bins flanking each
CBR, computes metaprofiles (mean ± SEM across CBRs), and decomposes the signal
by partner chromosome to show which chromosomes contribute to the interaction
enrichment near break sites.

Outputs:
  - Combined metaprofile across all CBRs (end-type and internal separately)
  - Per-CBR individual profiles and data tables
  - Grid overview of all CBRs
  - Component breakdown showing per-chromosome contributions (stacked area)
  - Selected sample comparisons with optional derivative curves

Dependencies:
  numpy, pandas, matplotlib

Example usage:
  python trans_cbr_profile.py \\
      --pre_bed data/AG_v50_chrom.bed \\
      --pre_samples allValidPairs/AG.v50.teste.trans.allValidPairs \\
                    allValidPairs/AG.v50.48hr.trans.allValidPairs \\
                    allValidPairs/AG.v50.60hr.trans.allValidPairs \\
      --post_samples allValidPairs/AG.v50.5day.trans.allValidPairs \\
      --bed data/cbr_v50_500kb_windows_cbr.bed \\
      --binsize 5000 --flank 500000 \\
      --normalize \\
      --components teste --component_type end \\
      --output_dir results/ \\
      --dpi 300

Input files:
  --pre_bed: Chromosome sizes/regions BED file for the pre-PDE genome.
  --pre_samples / --post_samples: Trans-only allValidPairs files from HiC-Pro
      (column 2 = query chr, column 5 = target chr).  These are large files
      NOT included in the repo; regenerate from SRA reads via HiC-Pro.
  --bed: BED file defining CBR positions with columns:
      chr  start  end  orientation  name  type (end/internal).
"""

import argparse
import os
import copy
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##############################################
# Debug helper
##############################################
def debug_print(msg, debug=False):
    if debug:
        print(f"[DEBUG] {msg}")

##############################################
# Parse sample name for legend labels.
##############################################
def parse_sample_label(sample_name):
    token = "AG.v50."
    idx = sample_name.find(token)
    if idx != -1:
        sub = sample_name[idx+len(token):]
        period_idx = sub.find('.')
        if period_idx != -1:
            return sub[:period_idx]
        return sub
    return sample_name

##############################################
# Save plot in multiple formats
##############################################
def save_plot_formats(base_path, dpi=600):
    """Save plot in PNG, TIFF, and SVG formats."""
    # Save PNG
    plt.savefig(base_path, dpi=dpi)
    
    # Save TIFF (without compression parameter which isn't supported)
    if base_path.endswith(".png"):
        tiff_path = base_path[:-4] + ".tiff"
        plt.savefig(tiff_path, dpi=dpi, format='tiff')
        print(f"TIFF version saved to {tiff_path}")
        
        # Save SVG
        svg_path = base_path[:-4] + ".svg"
        plt.savefig(svg_path, format='svg')
        print(f"SVG version saved to {svg_path}")

##############################################
# Plot combined sample profiles (for summary).
##############################################
def plot_all_sample_profiles(rel_centers, combined_profiles, combined_errors, dpi, ylim, output_plot, title="Combined Trans Interaction Profiles Centered on CBRs"):
    x = rel_centers / 1000.0
    mask = (rel_centers >= -500000) & (rel_centers <= 100000)
    x = x[mask]

    plt.figure(figsize=(12, 8))  # Increased width for legend outside
    cmap = plt.get_cmap('viridis')
    sample_names = list(combined_profiles.keys())
    n_samples = len(sample_names)
    for i, s in enumerate(sample_names):
        base_name = s.split(" (")[0]
        label = parse_sample_label(base_name)
        color = cmap(i / (n_samples - 1) if n_samples > 1 else 0.5)
        y = combined_profiles[s][mask]
        plt.plot(x, y, linestyle="-", linewidth=1.0, color=color, label=label)
        if combined_errors and s in combined_errors:
            plt.fill_between(x, combined_profiles[s][mask] - combined_errors[s][mask],
                             combined_profiles[s][mask] + combined_errors[s][mask], color=color, alpha=0.2)

    # Add vertical line and more visible shading at x=0 ±5kb
    plt.axvline(x=0, linestyle="--", color="black", linewidth=0.8)
    plt.axvspan(-5, 5, color="green", alpha=0.25, label="CBR region (±5kb)")  # Increased alpha from 0.08 to 0.25

    plt.xlabel("Relative Position (kb)")
    plt.ylabel("Hi-C Interaction Counts")
    if ylim is not None:
        plt.ylim(0, ylim)
    plt.title(title, fontsize=14)
    # Legend outside plot area to avoid overlap
    plt.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    save_plot_formats(output_plot, dpi)
    plt.close()
    print(f"Combined sample profiles plot saved to {output_plot}")

##############################################
# Load bins from a BED file (for the pre-PDE genome).
##############################################
def load_chrom_bins(bed_file, bin_size, ignore_chroms):
    chrom_bins = {}
    with open(bed_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            chrom = parts[0]
            if any(ig in chrom for ig in ignore_chroms):
                continue
            start = int(parts[1]) - 1
            end = int(parts[2])
            bins = {pos: 0 for pos in range(start, end, bin_size)}
            chrom_bins[chrom] = bins
    return chrom_bins

##############################################
# Process trans interactions from an allValidPairs file.
# Also track partner chromosomes for each bin
##############################################
def process_trans_pairs_with_partners(pairs_file, chrom_bins, bin_size, ignore_chroms):
    """Process trans pairs and track partner chromosomes for each bin."""
    # Initialize partner tracking
    bin_partners = {}
    for chrom in chrom_bins:
        bin_partners[chrom] = {}
        for bin_pos in chrom_bins[chrom]:
            bin_partners[chrom][bin_pos] = {}
    
    with open(pairs_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            chrA, chrB = parts[1], parts[4]
            if chrA == chrB:
                continue
            if any(ig in chrA for ig in ignore_chroms) or any(ig in chrB for ig in ignore_chroms):
                continue
            try:
                posA = int(parts[2]); posB = int(parts[5])
            except ValueError:
                continue
            binA = (posA // bin_size) * bin_size
            binB = (posB // bin_size) * bin_size
            
            # Process A -> B interaction
            if chrA in chrom_bins and binA in chrom_bins[chrA]:
                chrom_bins[chrA][binA] += 1
                if chrB not in bin_partners[chrA][binA]:
                    bin_partners[chrA][binA][chrB] = 0
                bin_partners[chrA][binA][chrB] += 1
            
            # Process B -> A interaction
            if chrB in chrom_bins and binB in chrom_bins[chrB]:
                chrom_bins[chrB][binB] += 1
                if chrA not in bin_partners[chrB][binB]:
                    bin_partners[chrB][binB][chrA] = 0
                bin_partners[chrB][binB][chrA] += 1
    
    return bin_partners

##############################################
# Process trans interactions from an allValidPairs file (original).
##############################################
def process_trans_pairs(pairs_file, chrom_bins, bin_size, ignore_chroms):
    with open(pairs_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            chrA, chrB = parts[1], parts[4]
            if chrA == chrB:
                continue
            if any(ig in chrA for ig in ignore_chroms) or any(ig in chrB for ig in ignore_chroms):
                continue
            try:
                posA = int(parts[2]); posB = int(parts[5])
            except ValueError:
                continue
            binA = (posA // bin_size) * bin_size
            binB = (posB // bin_size) * bin_size
            if chrA in chrom_bins and binA in chrom_bins[chrA]:
                chrom_bins[chrA][binA] += 1
            if chrB in chrom_bins and binB in chrom_bins[chrB]:
                chrom_bins[chrB][binB] += 1

##############################################
# Combine sample counts into a DataFrame.
##############################################
def combine_sample_counts(sample_counts, bin_size):
    rows = []
    sample_names = list(sample_counts.keys())
    chrom_set = set()
    for s in sample_names:
        chrom_set.update(sample_counts[s].keys())
    for chrom in sorted(chrom_set):
        bins_set = set()
        for s in sample_names:
            if chrom in sample_counts[s]:
                bins_set.update(sample_counts[s][chrom].keys())
        for b_start in sorted(bins_set):
            row = {"chromosome": chrom, "start": b_start, "end": b_start + bin_size}
            for s in sample_names:
                row[f"{s}_count"] = sample_counts[s].get(chrom, {}).get(b_start, 0)
            rows.append(row)
    return pd.DataFrame(rows)

##############################################
# Normalize counts to CPM.
##############################################
def normalize_df(df, sample_names, scale=1e6):
    for s in sample_names:
        col = f"{s}_count"
        norm_col = f"{s}_norm"
        total = df[col].sum()
        if total > 0:
            df[norm_col] = (df[col] / total) * scale
        else:
            df[norm_col] = df[col]
    return df

##############################################
# Write DataFrame to text file.
##############################################
def write_table_df(df, output_file):
    df.to_csv(output_file, sep="\t", index=False)
    print(f"Combined counts table written to {output_file}")

##############################################
# Read BED file for summary (ignoring chrUn, chrM).
##############################################
def read_bed(bed_file):
    regions = []
    with open(bed_file, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            chrom = parts[0]
            try:
                start = int(parts[1]); end = int(parts[2])
            except ValueError:
                continue
            if "chrUn" in chrom or "chrM" in chrom:
                continue
            center = (start + end) // 2
            orientation = parts[3].strip()
            region_name = parts[4].strip()
            region_type = parts[5].strip()
            regions.append((chrom, center, orientation, region_name, region_type))
    return regions

##############################################
# Read BED for individual trace (columns 1-5).
##############################################
def read_bed_for_traces(bed_file):
    regions = []
    with open(bed_file, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 6:
                chrom = parts[0]
                try:
                    start = int(parts[1]); end = int(parts[2])
                except ValueError:
                    continue
                center = (start + end) // 2
                orientation = parts[3].strip()
                region_name = parts[4].strip()
                regions.append((chrom, center, orientation, region_name))
            elif len(parts) >= 4:
                chrom = parts[0]
                try:
                    start = int(parts[1]); end = int(parts[2])
                except ValueError:
                    continue
                center = (start + end) // 2
                orientation = parts[3].strip()
                regions.append((chrom, center, orientation, orientation))
    return regions

##############################################
# Compute metaprofile from a list of regions.
##############################################
def compute_metaprofile_from_regions(df, regions, flank, bin_size, sample_names, debug=False):
    rel_edges = np.arange(-flank, flank + bin_size, bin_size)
    rel_centers = (rel_edges[:-1] + rel_edges[1:]) / 2.0
    profile_lists = {s: [] for s in sample_names}
    for (chrom, center, orient, *_ ) in regions:
        subdf = df[df["chromosome"] == chrom]
        if subdf.empty:
            debug_print(f"No data for region on {chrom} at center {center}", debug)
            continue
        subdf = subdf[(subdf["start"] + bin_size/2 >= center - flank) &
                      (subdf["start"] + bin_size/2 < center + flank)]
        if subdf.empty:
            debug_print(f"Region on {chrom} at {center} ({orient}): no bins in window", debug)
            continue
        debug_print(f"Region on {chrom} at {center} ({orient}): {subdf.shape[0]} bins selected", debug)

        temp_profiles = {s: np.full(len(rel_centers), np.nan, dtype=float) for s in sample_names}
        for idx, row in subdf.iterrows():
            rel_pos = (row["start"] + bin_size/2) - center
            if orient.upper() == "ER":
                rel_pos = -rel_pos
            bin_idx = np.digitize(rel_pos, rel_edges) - 1
            if 0 <= bin_idx < len(rel_centers):
                for s in sample_names:
                    col = f"{s}_norm" if f"{s}_norm" in subdf.columns else f"{s}_count"
                    if np.isnan(temp_profiles[s][bin_idx]):
                        temp_profiles[s][bin_idx] = row[col]
                    else:
                        temp_profiles[s][bin_idx] += row[col]
        for s in sample_names:
            profile_lists[s].append(temp_profiles[s])

    profile_means = {}
    profile_errors = {}
    for s in sample_names:
        if len(profile_lists[s]) == 0:
            profile_means[s] = np.full(len(rel_centers), np.nan)
            profile_errors[s] = np.full(len(rel_centers), np.nan)
        else:
            arr = np.vstack(profile_lists[s])
            counts = np.sum(~np.isnan(arr), axis=0)
            profile_means[s] = np.nanmean(arr, axis=0)
            errs = np.nanstd(arr, axis=0)
            errs[counts>0] /= np.sqrt(counts[counts>0])
            profile_errors[s] = errs

    return rel_centers, profile_means, profile_errors

##############################################
# Compute individual trace profile for one region.
##############################################
def compute_trace_profile(df, region, flank, bin_size, sample_names, debug=False):
    if len(region) >= 4:
        chrom, center, orient, region_name = region[:4]
    else:
        chrom, center, region_name = region[:3]
        orient = ""
    rel_edges = np.arange(-flank, flank + bin_size, bin_size)
    rel_centers = (rel_edges[:-1] + rel_edges[1:]) / 2.0

    subdf = df[df["chromosome"] == chrom]
    if subdf.empty:
        debug_print(f"No data for region {region_name} on {chrom} at center {center}", debug)
        profiles = {s: np.full(len(rel_centers), np.nan, dtype=float) for s in sample_names}
        return rel_centers, profiles

    subdf = subdf[(subdf["start"] + bin_size/2 >= center - flank) &
                  (subdf["start"] + bin_size/2 < center + flank)]
    if subdf.empty:
        debug_print(f"Region {region_name} on {chrom} at {center}: no bins in window", debug)
        profiles = {s: np.full(len(rel_centers), np.nan, dtype=float) for s in sample_names}
        return rel_centers, profiles

    profiles = {s: np.full(len(rel_centers), np.nan, dtype=float) for s in sample_names}
    for idx, row in subdf.iterrows():
        rel_pos = (row["start"] + bin_size/2) - center
        if orient.upper() == "ER":
            rel_pos = -rel_pos
        bin_idx = np.digitize(rel_pos, rel_edges) - 1
        if 0 <= bin_idx < len(rel_centers):
            for s in sample_names:
                col = f"{s}_norm" if f"{s}_norm" in df.columns else f"{s}_count"
                if np.isnan(profiles[s][bin_idx]):
                    profiles[s][bin_idx] = row[col]
                else:
                    profiles[s][bin_idx] += row[col]
    return rel_centers, profiles

##############################################
# Write individual trace data to text file.
##############################################
def write_trace_data(rel_centers, profiles, sample_names, output_file):
    x = rel_centers / 1000.0
    header = ["Relative_Position_kb"] + [parse_sample_label(s) for s in sample_names]
    lines = ["\t".join(header)]
    for i, xv in enumerate(x):
        row = [f"{xv:.3f}"]
        for s in sample_names:
            val = profiles[s][i]
            row.append(f"{val:.3f}" if not np.isnan(val) else "NaN")
        lines.append("\t".join(row))
    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    print(f"Trace data written to {output_file}")

##############################################
# Plot individual trace profile.
##############################################
def plot_trace_profile(rel_centers, profiles, region, dpi, ylim, output_file):
    x = rel_centers / 1000.0
    plt.figure(figsize=(8,6))
    cmap = plt.get_cmap('viridis')
    sample_names = list(profiles.keys())
    n_samples = len(sample_names)
    for i, s in enumerate(sample_names):
        label = parse_sample_label(s.split(" (")[0])
        color = cmap(i/(n_samples-1) if n_samples>1 else 0.5)
        plt.plot(x, profiles[s], linestyle='-', linewidth=1.0, color=color, label=label)

    plt.xlabel("Relative Position (kb)")
    plt.ylabel("Hi-C Interaction Counts")
    if ylim is not None:
        plt.ylim(0, ylim)

    plt.xlim(x[0], x[-1])
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)

    region_name = region[3] if len(region)>=4 else region[-1]
    plt.title(f"Region {region_name} on {region[0]} at {region[1]} (trans)", fontsize=14)
    plt.legend(fontsize=8, loc='best')
    plt.tight_layout()
    
    save_plot_formats(output_file, dpi)
    plt.close()
    print(f"Trace plot for region {region_name} saved to {output_file}")

##############################################
# Plot grid of individual trace profiles.
##############################################
def plot_trace_grid(df, regions, flank, bin_size, sample_names, dpi, ylim, output_file, debug=False):
    n_regions = len(regions)
    ncols = 12
    nrows = math.ceil(n_regions / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3), dpi=dpi)
    axes = axes.flatten()
    cmap = plt.get_cmap('viridis')
    for i, region in enumerate(regions):
        rel_centers, profiles = compute_trace_profile(df, region, flank, bin_size, sample_names, debug=debug)
        x = rel_centers / 1000.0
        ax = axes[i]
        for j, s in enumerate(sample_names):
            color = cmap(j/(len(sample_names)-1) if len(sample_names)>1 else 0.5)
            ax.plot(x, profiles[s], linestyle='-', linewidth=0.8, color=color)
        ax.set_title(region[3] + " (trans)", fontsize=8)
        ax.set_xlabel("kb", fontsize=6)
        ax.set_ylabel("Counts", fontsize=6)
        if ylim is not None:
            ax.set_ylim(0, ylim)
        ax.set_xlim(x[0], x[-1])
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)

    for k in range(n_regions, len(axes)):
        axes[k].axis('off')
    plt.tight_layout()
    
    save_plot_formats(output_file, dpi)
    plt.close()
    print(f"Grid plot of individual CBR traces saved to {output_file}")

##############################################
# Compute component profile using DataFrame data (NEW)
##############################################
def compute_component_profile_from_df(df, bin_partners, region, flank, bin_size, sample_name, debug=False):
    """Compute component profile using the processed DataFrame and partner tracking."""
    if len(region) >= 5:
        chrom, center, orient, region_name, _ = region
    else:
        chrom, center, orient, region_name = region
    
    edges = np.arange(-flank, flank + bin_size, bin_size)
    rel_centers = (edges[:-1] + edges[1:]) / 2.0
    comp_dict = {}
    
    # Get the sample column (normalized or count)
    col = f"{sample_name}_norm" if f"{sample_name}_norm" in df.columns else f"{sample_name}_count"
    
    # Get subset of data for this chromosome
    subdf = df[df["chromosome"] == chrom]
    if subdf.empty:
        debug_print(f"No data for region {region_name} on {chrom}", debug)
        return rel_centers, comp_dict
    
    # Get bins within window
    subdf = subdf[(subdf["start"] + bin_size/2 >= center - flank) &
                  (subdf["start"] + bin_size/2 < center + flank)]
    
    if subdf.empty:
        debug_print(f"Region {region_name}: no bins in window", debug)
        return rel_centers, comp_dict
    
    # Process each bin
    for idx, row in subdf.iterrows():
        bin_start = row["start"]
        rel_pos = (bin_start + bin_size/2) - center
        if orient.upper() == "ER":
            rel_pos = -rel_pos
        
        if -flank <= rel_pos < flank:
            bin_idx = int((rel_pos + flank) // bin_size)
            
            # Get the value for this bin
            value = row[col]
            if value > 0 and chrom in bin_partners and bin_start in bin_partners[chrom]:
                # Distribute the value proportionally among partner chromosomes
                partners = bin_partners[chrom][bin_start]
                total_partner_count = sum(partners.values())
                
                for partner, count in partners.items():
                    if "chrUn" in partner or "chrM" in partner:
                        continue
                    
                    # Proportional allocation
                    proportion = count / total_partner_count if total_partner_count > 0 else 0
                    allocated_value = value * proportion
                    
                    if partner not in comp_dict:
                        comp_dict[partner] = np.zeros(len(rel_centers))
                    comp_dict[partner][bin_idx] += allocated_value
    
    if debug:
        debug_print(f"Component profile for {region_name}: {len(comp_dict)} partners", debug)
    
    return rel_centers, comp_dict

##############################################
# Compute component metaprofile over multiple regions using DataFrame (NEW)
##############################################
def compute_component_metaprofile_from_df(df, bin_partners, regions, flank, bin_size, sample_name, debug=False):
    """Compute averaged component metaprofile using DataFrame."""
    all_components = {}
    rel_centers_final = None
    
    for region in regions:
        rel_centers, comp_dict = compute_component_profile_from_df(
            df, bin_partners, region, flank, bin_size, sample_name, debug
        )
        
        if rel_centers_final is None:
            rel_centers_final = rel_centers
        
        for partner, counts in comp_dict.items():
            if partner not in all_components:
                all_components[partner] = []
            all_components[partner].append(counts)
    
    # Average across regions
    avg_components = {}
    for partner, profiles in all_components.items():
        if len(profiles) > 0:
            avg_components[partner] = np.mean(np.vstack(profiles), axis=0)
    
    return rel_centers_final, avg_components

##############################################
# Filter regions by type for component analysis
##############################################
def filter_regions_by_type(regions, component_type):
    """Filter regions based on component_type parameter."""
    if component_type.lower() == 'all':
        return regions
    elif component_type.lower() == 'end':
        return [r for r in regions if len(r) >= 5 and r[4].lower() == 'end']
    elif component_type.lower() == 'internal':
        return [r for r in regions if len(r) >= 5 and r[4].lower() == 'internal']
    else:
        print(f"Warning: Unknown component_type '{component_type}', using all regions")
        return regions

##############################################
# Plot component breakdown for an individual region.
##############################################
def plot_component_profile(rel_centers, comp_dict, region, dpi, ylim, output_file):
    x = rel_centers / 1000.0
    partners = sorted(comp_dict.keys())
    data = [comp_dict[pt] for pt in partners]
    plt.figure(figsize=(10,6))  # Wider for legend
    plt.stackplot(x, *data, labels=partners)
    plt.xlabel("Relative Position (kb)")
    plt.ylabel("Interaction Counts")
    if ylim is not None:
        plt.ylim(0, ylim)
    if len(region) == 4:
        region_name = region[3]
    else:
        region_name = region[-1]
    plt.title(f"Component Breakdown for Region {region_name} (trans)", fontsize=14)
    # Legend outside to avoid overlap
    plt.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    save_plot_formats(output_file, dpi)
    plt.close()
    print(f"Component plot for region {region_name} saved to {output_file}")

##############################################
# Plot combined component metaprofile (averaged over all regions).
##############################################
def plot_component_metaprofile(rel_centers, comp_dict, dpi, ylim, output_file, sample_label, component_type="all"):
    x = rel_centers / 1000.0
    mask = (rel_centers >= -500000) & (rel_centers <= 100000)
    x_masked = x[mask]
    
    partners = sorted(comp_dict.keys())
    data = [comp_dict[pt][mask] for pt in partners]
    plt.figure(figsize=(12,8))  # Wider for legend outside
    plt.stackplot(x_masked, *data, labels=partners)
    
    # More visible vertical line and shading at x=0 ±5kb
    plt.axvline(x=0, linestyle="--", color="black", linewidth=0.8)
    plt.axvspan(-5, 5, color="green", alpha=0.25, label="CBR region (±5kb)")
    
    plt.xlabel("Relative Position (kb)")
    plt.ylabel("Interaction Counts")
    if ylim is not None:
        plt.ylim(0, ylim)
    
    # Add component type to title
    type_str = f" ({component_type.capitalize()} CBRs)" if component_type != "all" else ""
    plt.title(f"Combined Component Breakdown for {sample_label}{type_str}", fontsize=14)
    
    # Legend outside plot area
    plt.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    save_plot_formats(output_file, dpi)
    plt.close()
    print(f"Combined component plot saved to {output_file}")

##############################################
# Write STAR Methods description.
##############################################
def write_star_methods(output_dir):
    star_text = (
        "STAR Methods Description:\n"
        "---------------------------\n"
        "This study integrates trans Hi-C interaction data from two developmental stages (pre-PDE and post-PDE) \n"
        "to characterize the interaction landscape around chromatin boundary regions (CBRs). A full-genome chromosome \n"
        "sizes file (converted from 1-based to 0-based coordinates) was used to generate fixed-size bins, establishing \n"
        "a common coordinate framework for all samples. Trans interactions (i.e., between different chromosomes) were \n"
        "extracted from allValidPairs files, and counts were aggregated across samples into a single combined table. \n"
        "\n"
        "For each CBR, defined in a BED file with orientation (RE or ER), bins whose midpoints fall within a ±flank \n"
        "window around the CBR center were selected. For regions with an 'ER' orientation, relative positions were flipped \n"
        "to align with an 'RE' orientation. Metaprofiles were computed for each sample and overlaid in a combined plot \n"
        "to facilitate direct comparisons between pre-PDE and post-PDE states. Additionally, individual trace plots \n"
        "were generated for each CBR and arranged in a grid for comprehensive visualization. Separate summary plots \n"
        "were produced for CBRs of type 'end' and 'internal'.\n"
        "\n"
        "Selected sample plots were also generated for a user-defined subset of samples, with first derivative curves \n"
        "overlaid (in green) for designated samples. Each region's selected data (including derivative values) is saved to \n"
        "an individual text file. \n"
        "\n"
        "Component breakdown analysis uses the same normalized data as the main profiles to ensure consistency. \n"
        "For the designated sample, partner chromosome contributions are tracked during initial processing and then \n"
        "proportionally allocated based on the normalized interaction values. This ensures that component plots \n"
        "accurately reflect the same data processing pipeline as the overall profiles. Component analysis can be \n"
        "filtered by CBR type (end, internal, or all) using the --component_type parameter. A summary table of \n"
        "component breakdowns (with overall counts per partner) is provided as a text file, and a combined component \n"
        "summary plot is generated with partner names in alphabetical order.\n"
        "\n"
        "All figures are saved in multiple formats (PNG, TIFF, SVG) for publication purposes.\n"
        "\n"
        "Mermaid Diagram:\n"
        "```mermaid\n"
        "flowchart TD\n"
        "    A[Start: Hi-C allValidPairs files] --> B[Load pre-PDE BED and generate common bins]\n"
        "    B --> C[Process pre-PDE and post-PDE trans interactions]\n"
        "    C --> C2[Track partner chromosomes for each bin]\n"
        "    C2 --> D[Combine counts into a single table]\n"
        "    D --> D2[Apply normalization if specified]\n"
        "    D2 --> E[Extract CBR regions from BED file]\n"
        "    E --> F[Filter regions by type for additional summary plots]\n"
        "    E --> G[For each CBR, select bins within ±flank and flip for 'ER']\n"
        "    G --> H[Compute per-sample metaprofiles]\n"
        "    H --> I[Overlay all sample profiles in a combined plot]\n"
        "    H --> J[Generate individual trace plots per CBR]\n"
        "    J --> K[Arrange individual plots in a grid]\n"
        "    H --> L[Filter for selected samples and compute derivative curves]\n"
        "    L --> M[Generate selected sample plots + text files]\n"
        "    D2 --> N[For designated sample, use normalized data for components]\n"
        "    N --> N2[Filter regions by --component_type if specified]\n"
        "    N2 --> O[Generate component stacked area plots using same normalization]\n"
        "    M & O --> P[Write STAR Methods documentation]\n"
        "    P --> Q[End: Output in PNG, TIFF, and SVG formats]\n"
        "```\n"
    )
    out_file = os.path.join(output_dir, "STAR_methods.txt")
    with open(out_file, "w") as f:
        f.write(star_text)
    print(f"STAR Methods description written to {output_dir}/STAR_methods.txt")

##############################################
# Main function
##############################################
def main():
    parser = argparse.ArgumentParser(
        description="Compute and plot trans interaction profiles centered on CBRs for pre-PDE and post-PDE samples."
    )
    parser.add_argument("--pre_bed", required=True,
                        help="Pre-PDE BED file (chrom, start, end; 1-based) for chromosome sizes.")
    parser.add_argument("--pre_samples", required=True, nargs='+',
                        help="Paths to allValidPairs files for pre-PDE trans interactions.")
    parser.add_argument("--post_samples", required=True, nargs='+',
                        help="Paths to allValidPairs files for post-PDE trans interactions.")
    parser.add_argument("--bed", required=True,
                        help="BED file for CBRs and trace regions (with at least 6 columns).")
    parser.add_argument("--output_table", required=True,
                        help="Name of the combined counts table file (txt).")
    parser.add_argument("--output_plot", required=True,
                        help="Name of the combined metaprofile plot file (png).")
    parser.add_argument("--output_plot_end", required=True,
                        help="Name of the summary metaprofile plot for 'end' CBRs (png).")
    parser.add_argument("--output_plot_internal", required=True,
                        help="Name of the summary metaprofile plot for 'internal' CBRs (png).")
    parser.add_argument("--grid_plot", required=True,
                        help="Name of the grid plot of individual CBR traces (png).")
    parser.add_argument("--binsize", type=int, default=5000,
                        help="Bin size in bp (default: 5000).")
    parser.add_argument("--flank", type=int, default=250000,
                        help="Flank size in bp on either side of CBR center (default: 250000).")
    parser.add_argument("--dpi", type=int, default=600,
                        help="DPI for plots (default: 600).")
    parser.add_argument("--ylim", type=int, default=None,
                        help="Optional Y-axis upper limit.")
    parser.add_argument("--normalize", action="store_true",
                        help="Apply CPM normalization.")
    parser.add_argument("--scale", type=float, default=1e6,
                        help="Normalization scaling factor (default: 1e6).")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output (per region).")
    parser.add_argument("--output_dir", default="trans_profile",
                        help="Main output directory (default: trans_profile).")
    parser.add_argument("--selected_samples", nargs='+',
                        help="List of substrings for selected samples to include in separate plots.")
    parser.add_argument("--derivative_samples", nargs='+',
                        help="List of substrings (parsed sample names) for derivative curves.")
    parser.add_argument("--selected_dir", default="selected_cbrs",
                        help="Directory name (inside output_dir) for selected sample plots.")
    parser.add_argument("--components", help="Sample substring for which to generate component breakdown plots.")
    parser.add_argument("--component_type", default="all", choices=["end", "internal", "all"],
                        help="Type of CBRs to include in component analysis (default: all).")
    args = parser.parse_args()

    bin_size = args.binsize
    ignore_chroms = ["chrUn", "chrM"]
    os.makedirs(args.output_dir, exist_ok=True)

    # Build sample file map and process samples with partner tracking
    sample_file_map = {}
    sample_bin_partners = {}  # NEW: Track partner info for each sample
    pre_sample_counts = {}
    
    for fp in args.pre_samples:
        name = os.path.splitext(os.path.basename(fp))[0] + " (Pre)"
        parsed = parse_sample_label(name)
        sample_file_map[name] = fp
        sample_file_map[parsed] = fp
        bins_copy = copy.deepcopy(load_chrom_bins(args.pre_bed, bin_size, ignore_chroms))
        
        # Process with partner tracking for components
        bin_partners = process_trans_pairs_with_partners(fp, bins_copy, bin_size, ignore_chroms)
        sample_bin_partners[name] = bin_partners
        pre_sample_counts[name] = bins_copy

    post_sample_counts = {}
    for fp in args.post_samples:
        name = os.path.splitext(os.path.basename(fp))[0] + " (Post)"
        parsed = parse_sample_label(name)
        sample_file_map[name] = fp
        sample_file_map[parsed] = fp
        bins_copy = copy.deepcopy(load_chrom_bins(args.pre_bed, bin_size, ignore_chroms))
        
        # Process with partner tracking for components
        bin_partners = process_trans_pairs_with_partners(fp, bins_copy, bin_size, ignore_chroms)
        sample_bin_partners[name] = bin_partners
        post_sample_counts[name] = bins_copy

    combined_counts = {**pre_sample_counts, **post_sample_counts}
    df = combine_sample_counts(combined_counts, bin_size)
    all_samples = list(combined_counts.keys())
    if args.normalize:
        df = normalize_df(df, all_samples, scale=args.scale)

    # Write combined table
    write_table_df(df, os.path.join(args.output_dir, args.output_table))

    # Summary metaprofiles
    regions = read_bed(args.bed)
    summary_regions = [(c, ctr, ori, nm) for c, ctr, ori, nm, _ in regions]
    rc, pm, pe = compute_metaprofile_from_regions(df, summary_regions, args.flank, bin_size, all_samples, debug=args.debug)
    plot_all_sample_profiles(rc, pm, pe, args.dpi, args.ylim, os.path.join(args.output_dir, args.output_plot))

    # End/Internal splits
    end_r = [(c, ctr, ori, nm) for c, ctr, ori, nm, t in regions if t.lower()=='end']
    int_r = [(c, ctr, ori, nm) for c, ctr, ori, nm, t in regions if t.lower()=='internal']
    if end_r:
        rc_e, pm_e, pe_e = compute_metaprofile_from_regions(df, end_r, args.flank, bin_size, all_samples, debug=args.debug)
        plot_all_sample_profiles(rc_e, pm_e, pe_e, args.dpi, args.ylim, os.path.join(args.output_dir, args.output_plot_end),
                                 title="Combined Trans Interaction Profiles for End CBRs")
    if int_r:
        rc_i, pm_i, pe_i = compute_metaprofile_from_regions(df, int_r, args.flank, bin_size, all_samples, debug=args.debug)
        plot_all_sample_profiles(rc_i, pm_i, pe_i, args.dpi, args.ylim, os.path.join(args.output_dir, args.output_plot_internal),
                                 title="Combined Trans Interaction Profiles for Internal CBRs")

    # Individual traces
    trace_regions = read_bed_for_traces(args.bed)
    indiv_dir = os.path.join(args.output_dir, 'individual_cbrs')
    os.makedirs(indiv_dir, exist_ok=True)
    for region in trace_regions:
        relc, prof = compute_trace_profile(df, region, args.flank, bin_size, all_samples, debug=args.debug)
        out_png = os.path.join(indiv_dir, f"{region[3]}_trans.png")
        plot_trace_profile(relc, prof, region, args.dpi, args.ylim, out_png)
        out_txt = os.path.join(indiv_dir, f"{region[3]}_trans_data.txt")
        write_trace_data(relc, prof, all_samples, out_txt)

    # Grid of all traces
    grid_png = os.path.join(args.output_dir, args.grid_plot)
    plot_trace_grid(df, trace_regions, args.flank, bin_size, all_samples, args.dpi, args.ylim, grid_png, debug=args.debug)

    # Selected sample plots with derivative curves
    if args.selected_samples:
        selected_dir = os.path.join(args.output_dir, args.selected_dir)
        os.makedirs(selected_dir, exist_ok=True)
        selected_summary = []

        for region in trace_regions:
            if len(region) >= 4:
                region_name = region[3]
                chrom = region[0]
                center = region[1]
            else:
                region_name = region[-1]
                chrom = region[0]
                center = region[1]

            rel, profiles = compute_trace_profile(df, region, args.flank, bin_size, all_samples, debug=args.debug)
            selected_profiles = {}
            for s in profiles:
                label = parse_sample_label(s)
                for sel in args.selected_samples:
                    if sel in label:
                        selected_profiles[s] = profiles[s]
                        break
            if not selected_profiles:
                continue

            region_data_file = os.path.join(selected_dir, f"{region_name}_trans_selected_data.txt")
            with open(region_data_file, "w") as dataf:
                dataf.write("X_kb\tSample\tNormalValue\tDerivativeValue\n")

                x_vals = rel / 1000.0
                plt.figure(figsize=(8,6))
                cmap = plt.get_cmap('viridis')
                plot_samples = list(selected_profiles.keys())
                n_sel = len(plot_samples)
                deriv_samples = []

                for i, s in enumerate(plot_samples):
                    base_name = s.split(" (")[0]
                    label = parse_sample_label(base_name)
                    y_vals = selected_profiles[s]
                    add_deriv = False
                    if args.derivative_samples:
                        for der in args.derivative_samples:
                            if der in label:
                                add_deriv = True
                                break

                    color = "green" if add_deriv else cmap(i/(n_sel-1) if n_sel>1 else 0.5)
                    plt.plot(x_vals, y_vals, linestyle="-", linewidth=1.0, color=color, label=label if not add_deriv else (label + " (derivative)"))
                    deriv_vals = None
                    if add_deriv:
                        deriv_vals = np.gradient(y_vals, x_vals)
                        plt.plot(x_vals, deriv_vals, linestyle="--", linewidth=1.0, color="green", label=label + " derivative")
                        deriv_samples.append(label)

                    for idx_d in range(len(x_vals)):
                        dv = f"{deriv_vals[idx_d]:.3f}" if deriv_vals is not None else "None"
                        dataf.write(f"{x_vals[idx_d]:.3f}\t{label}\t{y_vals[idx_d]:.3f}\t{dv}\n")

                plt.xlabel("Relative Position (kb)")
                plt.ylabel("Hi-C Interaction Counts")
                if args.ylim is not None:
                    plt.ylim(0, args.ylim)
                plt.title(f"Region {region_name} on {chrom} at {center} (trans, selected)", fontsize=14)
                plt.legend(fontsize=8, loc="best")
                plt.tight_layout()
                selected_out = os.path.join(selected_dir, f"{region_name}_trans_selected.png")
                save_plot_formats(selected_out, args.dpi)
                plt.close()
                print(f"Selected sample trace plot for region {region_name} saved to {selected_out}")
                print(f"Selected sample data written to {region_data_file}")

            selected_str = ", ".join([parse_sample_label(s) for s in plot_samples])
            deriv_str = ", ".join(deriv_samples) if deriv_samples else "None"
            selected_summary.append([region_name, chrom, center, "?", selected_str, deriv_str])

        if selected_summary:
            summary_file = os.path.join(selected_dir, "selected_cbrs_summary.txt")
            summary_df = pd.DataFrame(selected_summary, columns=["Region_Name", "Chromosome", "Center", "Orientation", "Selected_Samples", "Derivative_Samples"])
            summary_df.to_csv(summary_file, sep="\t", index=False)
            print(f"Selected sample summary table written to {summary_file}")

    # Components: generate component breakdown using normalized DataFrame data
    if args.components:
        component_sample_name = None
        component_sample_label = None
        
        # Find the matching sample name
        for sample_name in all_samples:
            label = parse_sample_label(sample_name)
            if args.components in label:
                component_sample_name = sample_name
                component_sample_label = label
                break
        
        if component_sample_name is None:
            print(f"No sample found matching --components substring: {args.components}")
        else:
            components_dir = os.path.join(args.output_dir, "components")
            os.makedirs(components_dir, exist_ok=True)
            
            # Get the bin partners for this sample
            bin_partners = sample_bin_partners.get(component_sample_name, {})
            
            if not bin_partners:
                print(f"Warning: No partner tracking data for sample {component_sample_name}")
            else:
                # Filter regions by component_type
                filtered_regions = filter_regions_by_type(regions, args.component_type)
                print(f"Processing {len(filtered_regions)} regions for component analysis (type: {args.component_type})")
                
                comp_summary = []
                # Per-region component plots
                for region in trace_regions:
                    rel_centers_comp, comp_dict = compute_component_profile_from_df(
                        df, bin_partners, region, args.flank, bin_size, 
                        component_sample_name, debug=args.debug
                    )
                    
                    if comp_dict:  # Only plot if there's data
                        comp_out = os.path.join(components_dir, f"{region[3]}_trans_components.png")
                        plot_component_profile(rel_centers_comp, comp_dict, region, 
                                              dpi=args.dpi, ylim=args.ylim, output_file=comp_out)
                        comp_sums = {pt: int(np.sum(comp_dict[pt])) for pt in comp_dict}
                        region_name = region[3] if len(region)>=4 else "Unknown"
                        comp_summary.append([region_name, region[0], region[1], 
                                           region[2] if len(region)>=3 else "?", str(comp_sums)])
                
                if comp_summary:
                    comp_table = os.path.join(components_dir, f"components_summary_{args.component_type}.txt")
                    comp_df = pd.DataFrame(comp_summary, 
                                          columns=["Region_Name","Chromosome","Center","Orientation","Component_Sums"])
                    comp_df.to_csv(comp_table, sep="\t", index=False)
                    print(f"Components summary table written to {comp_table}")
                
                # Combined metaprofile plot with filtered regions
                rel_centers_comb, comp_profiles_comb = compute_component_metaprofile_from_df(
                    df, bin_partners, filtered_regions, args.flank, bin_size, 
                    component_sample_name, debug=args.debug
                )
                
                if rel_centers_comb is not None and comp_profiles_comb:
                    comp_comb_plot = os.path.join(components_dir, f"combined_components_{args.component_type}.png")
                    plot_component_metaprofile(rel_centers_comb, comp_profiles_comb, 
                                              dpi=args.dpi, ylim=args.ylim, 
                                              output_file=comp_comb_plot, 
                                              sample_label=component_sample_label, 
                                              component_type=args.component_type)

    # Write STAR Methods
    write_star_methods(args.output_dir)

if __name__ == '__main__':

    main()
