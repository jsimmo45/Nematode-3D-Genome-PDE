#!/usr/bin/env python
"""
plot_interaction_heatmap.py
===========================
Quantify and visualize Hi-C interaction frequencies between pairs of genomic
regions across developmental timepoints.  Designed for organisms that undergo
programmed DNA elimination (PDE), where a single germline chromosome fragments
into multiple somatic chromosomes.

Outputs:
  - Region × timepoint heatmap (PNG + SVG)
  - Dot + mean/median summary plot with optional Mann-Whitney U significance
    brackets (PNG + SVG)
  - Interaction score matrix (CSV)

Key features:
  - Chromosome-aware O/E normalization: per-chromosome expected values prevent
    contamination from inter-chromosomal zeros in the diagonal average.
  - Dual coordinate system support: germline bin mapping is always used for
    matrix lookups; germline-to-somatic mapping is used only to classify
    post-PDE region pairs as intra- vs inter-chromosomal for expected values.
  - log2(O/E) with symmetric diverging color scale (RdBu_r) for publication
    figures.
  - SVG output with editable text (plt.rcParams['svg.fonttype'] = 'none')
    for downstream editing in Adobe Illustrator.

Dependencies:
  numpy, pandas, matplotlib, seaborn, scipy

Example usage:
  # log2(O/E) normalization across 5 timepoints (recommended for publication):
  python plot_interaction_heatmap.py \\
      --regions data/interaction_regions/interacting_regions_100kb.bed \\
      --timepoints '10hr,17hr,24hr,48hr,72hr' \\
      --resolution 20000 \\
      --bin-bed data/20000/pu_v2_prepde_abs_20kb \\
      --soma-bin-bed data/20000/pu_v2_postpde_abs_20kb \\
      --germ-to-soma-mapping data/pu_v2_germ_to_soma_mapping.bed \\
      --oe-normalize \\
      --log2-oe \\
      --method mean \\
      --color-dots \\
      --ref-timepoint 17hr \\
      --output interaction_heatmap_100kb \\
      --output-dir results/

  # Raw interaction frequencies (no normalization):
  python plot_interaction_heatmap.py \\
      --regions data/interaction_regions/interacting_regions_100kb.bed \\
      --timepoints '10hr,17hr,24hr,48hr,72hr' \\
      --resolution 20000 \\
      --bin-bed data/20000/pu_v2_prepde_abs_20kb \\
      --no-normalize \\
      --output interaction_heatmap_raw \\
      --output-dir results/

Input files:
  --regions BED file: chr  start1  end1  start2  end2  [name]
      Pairs of genomic windows whose interaction frequency is scored.
  --bin-bed: HiC-Pro abs.bed file mapping bin numbers to germline coordinates.
  --soma-bin-bed: abs.bed for somatic genome (used only for postpde bin info).
  --germ-to-soma-mapping: Tab-delimited file mapping germline coordinate
      ranges to somatic chromosome names (germ_chr  start  end  soma_chr).
  Hi-C matrices: ICE-normalized sparse matrices from HiC-Pro, expected at
      matrix_files_{resolution}/prepde/ and matrix_files_{resolution}/postpde/.

Normalization notes:
  --oe-normalize computes expected contact frequency per genomic distance
  within each chromosome, then divides observed by expected.  --log2-oe
  applies log2 to the O/E ratio, centering enrichment/depletion around 0.
  --cpm is redundant with --oe-normalize (the CPM scalar cancels in the
  ratio) and will be ignored if both are set.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy.sparse import coo_matrix
import argparse
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from collections import defaultdict

# ---------------------------------------------------------------------------
# Global plot styling — Arial 6pt for publication figures
# ---------------------------------------------------------------------------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans']
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['font.size'] = 6
plt.rcParams['axes.labelsize'] = 6
plt.rcParams['axes.titlesize'] = 6
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6
plt.rcParams['legend.fontsize'] = 6
plt.rcParams['svg.fonttype'] = 'none'  # editable text in SVG for Illustrator


# ===========================================================================
#  I/O helpers
# ===========================================================================

def read_germ_to_soma_mapping(mapping_file):
    """Read germline-to-somatic chromosome coordinate mapping.

    File format (tab-separated):
        germ_chrom  germ_start  germ_end  soma_chrom

    Returns list of dicts or None if file missing.
    """
    if not os.path.exists(mapping_file):
        print(f"  Warning: Mapping file not found: {mapping_file}")
        return None

    mappings = []
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                mappings.append({
                    'germ_chrom': parts[0],
                    'germ_start': int(parts[1]),
                    'germ_end':   int(parts[2]),
                    'soma_chrom': parts[3],
                })

    print(f"  Loaded {len(mappings)} germline->somatic chromosome mappings")
    return mappings


def translate_germline_to_somatic(germ_chrom, germ_pos, germ_to_soma_mappings):
    """Translate germline coordinate to (soma_chrom, offset_within_soma_chrom).

    Returns original coordinate if no mapping found.
    """
    if not germ_to_soma_mappings:
        return germ_chrom, germ_pos

    for m in germ_to_soma_mappings:
        if (m['germ_chrom'] == germ_chrom and
                m['germ_start'] <= germ_pos <= m['germ_end']):
            return m['soma_chrom'], germ_pos - m['germ_start']

    return germ_chrom, germ_pos


def read_bin_mapping(bed_file, resolution=20000):
    """Read HiC-Pro bin mapping from *_abs.bed file.

    Returns:
        bin_to_coords   : dict  bin_num(1-based) -> (chrom, start, end)
        coord_to_bin    : dict  (chrom, start)   -> bin_index(0-based)
        chrom_info      : dict  chrom -> {'min_bin', 'max_bin'} (1-based)
        chrom_bin_index : dict  chrom -> sorted list of 0-based bin indices
            (pre-indexed for fast per-chromosome lookup)
    """
    bin_to_coords   = {}
    coord_to_bin    = {}
    chrom_info      = {}
    chrom_bin_index = defaultdict(list)

    with open(bed_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                chrom   = parts[0]
                start   = int(parts[1])
                end     = int(parts[2])
                bin_num = int(parts[3])
                bin_idx = bin_num - 1          # 0-based for matrix access

                bin_to_coords[bin_num]      = (chrom, start, end)
                coord_to_bin[(chrom, start)] = bin_idx
                chrom_bin_index[chrom].append(bin_idx)

                if chrom not in chrom_info:
                    chrom_info[chrom] = {'min_bin': bin_num, 'max_bin': bin_num}
                else:
                    chrom_info[chrom]['max_bin'] = bin_num

    # Ensure sorted (should already be, but defensive)
    for chrom in chrom_bin_index:
        chrom_bin_index[chrom].sort()

    print(f"  Chromosomes in bin mapping: {', '.join(chrom_info.keys())}")
    for chrom, info in chrom_info.items():
        print(f"    {chrom}: bins {info['min_bin']} to {info['max_bin']}")

    return bin_to_coords, coord_to_bin, chrom_info, chrom_bin_index


def read_interacting_regions(bed_file):
    """Read pairs of interacting regions from BED file.

    Expected columns: chr  start1  end1  start2  end2  [name]
    """
    regions = []
    with open(bed_file, 'r') as f:
        for idx, line in enumerate(f):
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                regions.append({
                    'chr':    parts[0],
                    'start1': int(parts[1]),
                    'end1':   int(parts[2]),
                    'start2': int(parts[3]),
                    'end2':   int(parts[4]),
                    'name':   parts[5] if len(parts) > 5 else f"Region_{idx+1}",
                })
    return regions


def read_hic_matrix(filepath, resolution=20000):
    """Read Hi-C sparse matrix (HiC-Pro format: bin_i  bin_j  value).

    Returns (dense_matrix, total_contacts) or (None, 0) on failure.
    """
    print(f"  Reading {filepath}...")
    if not os.path.exists(filepath):
        print(f"  Warning: File not found: {filepath}")
        return None, 0

    row_indices = []
    col_indices = []
    data = []

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) == 3:
                try:
                    i     = int(parts[0]) - 1
                    j     = int(parts[1]) - 1
                    value = float(parts[2])
                except (ValueError, IndexError):
                    print(f"    Skipping malformed line {line_num}: {line.strip()}")
                    continue

                row_indices.append(i)
                col_indices.append(j)
                data.append(value)
                if i != j:                     # symmetrise
                    row_indices.append(j)
                    col_indices.append(i)
                    data.append(value)

    if not data:
        return None, 0

    max_idx = max(max(row_indices), max(col_indices)) + 1
    sparse_matrix = coo_matrix((data, (row_indices, col_indices)),
                               shape=(max_idx, max_idx))
    dense_matrix = sparse_matrix.toarray()
    # Each off-diagonal contact was stored twice; total unique contacts:
    total_contacts = np.sum(data) / 2
    print(f"    Matrix shape: {dense_matrix.shape}, Total contacts: {total_contacts:.0f}")
    return dense_matrix, total_contacts


# ===========================================================================
#  Bin resolution helpers
# ===========================================================================

def _resolve_bins_for_chrom(start, end, chrom, coord_to_bin, resolution):
    """Resolve a genomic interval on one chromosome to (bin_start, bin_end).

    Uses coord_to_bin dict filtered to matching chromosome.
    Returns (bin_start, bin_end) where bin_end is exclusive, or (None, None).
    """
    if coord_to_bin is None:
        return None, None

    # Collect all bin entries for this chromosome, sorted by position
    chrom_entries = sorted(
        [(pos, idx) for (c, pos), idx in coord_to_bin.items() if c == chrom]
    )

    if not chrom_entries:
        return None, None

    bin_start = None
    bin_end   = None
    for pos, bin_idx in chrom_entries:
        if pos <= start:
            bin_start = bin_idx
        if pos < end:
            bin_end = bin_idx

    if bin_end is not None:
        bin_end += 1  # exclusive

    return bin_start, bin_end


def resolve_region_bins(region, coord_to_bin, resolution,
                        germ_to_soma_mappings=None):
    """Resolve a pair of genomic regions to matrix bin ranges.

    Handles germline -> somatic coordinate translation when needed.

    Returns dict with keys:
        bin_start1, bin_end1, bin_start2, bin_end2  (0-based, end exclusive)
        chrom1, chrom2   — resolved chromosome names (somatic if translated)
        same_chrom       — bool
    """
    if germ_to_soma_mappings:
        chrom1, sstart1 = translate_germline_to_somatic(
            region['chr'], region['start1'], germ_to_soma_mappings)
        _,      send1   = translate_germline_to_somatic(
            region['chr'], region['end1'], germ_to_soma_mappings)
        chrom2, sstart2 = translate_germline_to_somatic(
            region['chr'], region['start2'], germ_to_soma_mappings)
        _,      send2   = translate_germline_to_somatic(
            region['chr'], region['end2'], germ_to_soma_mappings)
    else:
        chrom1 = chrom2 = region['chr']
        sstart1, send1 = region['start1'], region['end1']
        sstart2, send2 = region['start2'], region['end2']

    bs1, be1 = _resolve_bins_for_chrom(sstart1, send1, chrom1,
                                       coord_to_bin, resolution)
    bs2, be2 = _resolve_bins_for_chrom(sstart2, send2, chrom2,
                                       coord_to_bin, resolution)

    # Fallback to simple coordinate / resolution if lookup failed
    if bs1 is None:
        bs1 = region['start1'] // resolution
        be1 = region['end1']   // resolution
    if bs2 is None:
        bs2 = region['start2'] // resolution
        be2 = region['end2']   // resolution

    return {
        'bin_start1': bs1, 'bin_end1': be1,
        'bin_start2': bs2, 'bin_end2': be2,
        'chrom1': chrom1, 'chrom2': chrom2,
        'same_chrom': (chrom1 == chrom2),
    }


# ===========================================================================
#  Somatic chromosome index from germline bins  (for post-PDE expected)
# ===========================================================================

def build_somatic_chrom_bin_index(coord_to_bin, germ_to_soma_mappings):
    """Group GERMLINE bin indices by somatic chromosome.

    All Hi-C matrices are in germline coordinate space.  For post-PDE expected
    calculation we need to know which germline bins belong to which somatic
    chromosome so we can compute intra-chromosome expected per somatic chrom.

    Returns dict: soma_chrom -> sorted list of 0-based germline bin indices
    """
    soma_bin_index = defaultdict(list)

    for (chrom, pos), bin_idx in coord_to_bin.items():
        if not germ_to_soma_mappings:
            soma_bin_index[chrom].append(bin_idx)
            continue

        soma_chrom = None
        for m in germ_to_soma_mappings:
            if (m['germ_chrom'] == chrom and
                    m['germ_start'] <= pos < m['germ_end']):
                soma_chrom = m['soma_chrom']
                break

        if soma_chrom:
            soma_bin_index[soma_chrom].append(bin_idx)
        # else: eliminated region or unplaced scaffold — skip

    for sc in soma_bin_index:
        soma_bin_index[sc].sort()

    print(f"    Mapped germline bins to {len(soma_bin_index)} somatic chromosomes")
    for sc in sorted(soma_bin_index.keys()):
        bins = soma_bin_index[sc]
        print(f"      {sc}: {len(bins)} bins (germline idx {bins[0]}-{bins[-1]})")

    return soma_bin_index


# ===========================================================================
#  Expected value calculation — CHROMOSOME-AWARE  (FIX #1)
# ===========================================================================

def calculate_expected_chrom_aware(matrix, bin_to_coords, chrom_bin_index,
                                   resolution=20000):
    """Compute expected contact frequency by genomic distance, per chromosome.

    Unlike the naïve whole-matrix diagonal approach, this only averages
    intra-chromosomal contacts at each distance, preventing inter-chromosomal
    zeros/noise from contaminating the expected profile.

    Also computes a single inter-chromosomal expected value for use with
    post-PDE region pairs that land on different somatic chromosomes.

    All bin values (including zeros) are included — this is the standard
    approach in Hi-C analysis.  ICE-filtered bins that are truly invalid
    should ideally be masked upstream.

    Returns:
        intra_expected : dict  distance_in_bins -> mean_contact
        inter_expected : float  mean inter-chromosomal contact
    """
    n = matrix.shape[0]
    chroms = list(chrom_bin_index.keys())
    print(f"    Computing chromosome-aware expected for {len(chroms)} chromosome(s)..."
          f" (matrix size: {n})")

    distance_sums   = defaultdict(float)
    distance_counts = defaultdict(int)

    for chrom in chroms:
        # Filter out bin indices that exceed matrix dimensions
        bins  = np.array([b for b in chrom_bin_index[chrom] if b < n])
        nbins = len(bins)
        if nbins == 0:
            continue

        # Extract intra-chromosomal submatrix
        submat = matrix[np.ix_(bins, bins)]

        for d in range(nbins):
            diag = np.diagonal(submat, offset=d)
            # Include ALL values (zeros too) — standard Hi-C expected
            distance_sums[d]   += np.sum(diag)
            distance_counts[d] += len(diag)

    intra_expected = {}
    for d in sorted(distance_sums.keys()):
        if distance_counts[d] > 0:
            intra_expected[d] = distance_sums[d] / distance_counts[d]
        else:
            intra_expected[d] = 0.0

    # Fallback for distances beyond any single chromosome
    if intra_expected:
        max_d = max(d for d in intra_expected if isinstance(d, int))
        intra_expected['default'] = intra_expected[max_d]
    else:
        intra_expected['default'] = 1.0

    # --- Inter-chromosomal expected ---
    if len(chroms) > 1:
        inter_sum   = 0.0
        inter_count = 0
        for ci in range(len(chroms)):
            for cj in range(ci + 1, len(chroms)):
                bins_i = np.array([b for b in chrom_bin_index[chroms[ci]] if b < n])
                bins_j = np.array([b for b in chrom_bin_index[chroms[cj]] if b < n])
                if len(bins_i) == 0 or len(bins_j) == 0:
                    continue
                submat = matrix[np.ix_(bins_i, bins_j)]
                inter_sum   += np.sum(submat)
                inter_count += submat.size
        inter_expected = inter_sum / inter_count if inter_count > 0 else 0.0
        print(f"    Inter-chromosomal expected: {inter_expected:.6f} "
              f"({inter_count} bin-pairs)")
    else:
        inter_expected = 0.0
        print(f"    Single chromosome — no inter-chromosomal expected computed")

    # Report sample intra-chromosomal expected values
    sample_dists = [0, 1, 5, 10, 50, 100, 500]
    for d in sample_dists:
        if d in intra_expected:
            print(f"    Expected at {d} bins ({d * resolution / 1e3:.0f} kb): "
                  f"{intra_expected[d]:.4f}")

    return intra_expected, inter_expected


# ===========================================================================
#  Interaction scoring
# ===========================================================================

def calculate_interaction_score(matrix, region, coord_to_bin, resolution,
                                method, germ_to_soma_mappings):
    """Calculate raw interaction score between two regions.

    NOTE: Only non-zero values contribute to the mean/percentile.  This is
    intentional for ICE-normalised matrices where 0 can indicate a filtered
    (low-mappability) bin rather than a true zero-contact bin.
    """
    if matrix is None:
        return np.nan

    info = resolve_region_bins(region, coord_to_bin, resolution,
                               germ_to_soma_mappings)

    bs1, be1 = info['bin_start1'], info['bin_end1']
    bs2, be2 = info['bin_start2'], info['bin_end2']

    max_bin = matrix.shape[0]
    bs1 = max(0, min(bs1, max_bin))
    be1 = max(0, min(be1, max_bin))
    bs2 = max(0, min(bs2, max_bin))
    be2 = max(0, min(be2, max_bin))

    if bs1 >= be1 or bs2 >= be2:
        return np.nan

    submatrix = matrix[bs1:be1, bs2:be2]
    contacts  = submatrix[submatrix > 0]

    if len(contacts) == 0:
        return 0.0

    if method == 'mean':
        return np.mean(contacts)
    elif method == 'max':
        return np.max(submatrix)
    elif method == 'sum':
        return np.sum(contacts)
    elif method == 'percentile95':
        return np.percentile(contacts, 95)
    else:
        return np.mean(contacts)


def calculate_oe_interaction_score(matrix, region, coord_to_bin, resolution,
                                   method, germ_to_soma_mappings,
                                   intra_expected, inter_expected, use_log2,
                                   same_chrom_mappings=None):
    """Calculate O/E-normalised interaction score.

    Parameters
    ----------
    germ_to_soma_mappings : used for bin lookup coordinate translation (None = germline)
    same_chrom_mappings   : used ONLY to determine if regions are on the same
                            somatic chromosome (for choosing intra vs inter expected).
                            If None, assumes all regions are on the same chromosome.
    """
    observed = calculate_interaction_score(
        matrix, region, coord_to_bin, resolution,
        method, germ_to_soma_mappings)

    if np.isnan(observed) or observed == 0:
        return observed

    # Determine same_chrom using separate mapping (not bin lookup mapping)
    if same_chrom_mappings:
        sc1, _ = translate_germline_to_somatic(
            region['chr'], (region['start1'] + region['end1']) // 2,
            same_chrom_mappings)
        sc2, _ = translate_germline_to_somatic(
            region['chr'], (region['start2'] + region['end2']) // 2,
            same_chrom_mappings)
        same_chrom = (sc1 == sc2)
    else:
        same_chrom = True  # prepde: everything on chrX

    if same_chrom:
        # Intra-chromosomal: use distance-based expected
        center1 = (region['start1'] + region['end1']) // 2
        center2 = (region['start2'] + region['end2']) // 2

        if same_chrom_mappings:
            # Use somatic-coordinate distance
            _, sd1 = translate_germline_to_somatic(
                region['chr'], center1, same_chrom_mappings)
            _, sd2 = translate_germline_to_somatic(
                region['chr'], center2, same_chrom_mappings)
            distance = abs(sd2 - sd1)
        else:
            distance = abs(center2 - center1)

        distance_bins = distance // resolution
        expected = intra_expected.get(distance_bins,
                                      intra_expected.get('default', 1.0))
    else:
        # Inter-chromosomal: flat expected
        expected = inter_expected if inter_expected > 0 else 1.0

    if expected > 0:
        oe_score = observed / expected
    else:
        oe_score = observed

    if use_log2 and oe_score > 0:
        oe_score = np.log2(oe_score)

    return oe_score


# ===========================================================================
#  Plotting
# ===========================================================================

def _build_ylabel(log2_oe, oe_normalize):
    """Return the appropriate y-axis label for the current normalization."""
    if log2_oe and oe_normalize:
        return r'log$_2$(O/E)'
    elif oe_normalize:
        return 'O/E'
    else:
        return 'Interaction Frequency'


def _style_summary_ax(ax, timepoint_positions, timepoint_labels, ylabel,
                      ymax=None):
    """Apply shared styling to summary plot axes."""
    ax.set_xlabel('Developmental Timepoint (Hours)')
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='both', which='major', width=0.5, length=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.set_xticks(timepoint_positions)
    ax.set_xticklabels(timepoint_labels)
    if ymax is not None:
        ax.set_ylim(top=ymax * 1.05)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)


def _pval_to_stars(pval):
    """Convert p-value to significance stars."""
    if pval <= 0.0001:
        return '****'
    elif pval <= 0.001:
        return '***'
    elif pval <= 0.01:
        return '**'
    elif pval <= 0.05:
        return '*'
    else:
        return 'ns'


def create_summary_plot(interaction_matrix, regions, timepoints,
                        output_prefix='interaction_summary',
                        output_dir='parascaris_interaction_heatmaps',
                        ymax=None, oe_normalize=False, log2_oe=False,
                        summary_stat='both', color_dots=False,
                        dot_vmin=None, dot_vmax=None,
                        ref_timepoint=None):
    """Dot + mean/median + SEM summary plot across developmental timepoints.

    Parameters
    ----------
    summary_stat : str
        'mean', 'median', or 'both'.  When 'both', two separate plots are
        saved (one per stat) plus one combined plot.
    color_dots : bool
        If True, color individual dots using RdBu_r colormap matched to the
        heatmap scale. If False (default), dots are black.
    dot_vmin, dot_vmax : float or None
        Color scale limits for colored dots. If None, auto-computed from data.
    ref_timepoint : str or None
        Timepoint to use as reference for Mann-Whitney U tests (e.g. '17hr').
        Compared against all other timepoints. If None, no stats shown.
    """
    from scipy.stats import mannwhitneyu

    # --- Resolve timepoint positions and labels ---
    timepoint_labels    = []
    timepoint_positions = []
    for i, tp in enumerate(timepoints):
        if 'hr' in tp:
            label = tp.replace('hr', '')
            timepoint_labels.append(label)
            try:
                timepoint_positions.append(float(label))
            except ValueError:
                timepoint_positions.append(i)
        else:
            timepoint_labels.append(tp)
            timepoint_positions.append(i)

    ylabel = _build_ylabel(log2_oe, oe_normalize)

    # --- Colormap setup for colored dots ---
    if color_dots:
        from matplotlib.colors import Normalize as mplNormalize
        if dot_vmin is None:
            dot_vmin = np.nanmin(interaction_matrix)
        if dot_vmax is None:
            dot_vmax = np.nanmax(interaction_matrix)
        # Symmetric for log2(O/E)
        if log2_oe and oe_normalize:
            abs_max = max(abs(dot_vmin), abs(dot_vmax))
            dot_vmin, dot_vmax = -abs_max, abs_max
        dot_norm = mplNormalize(vmin=dot_vmin, vmax=dot_vmax)
        dot_cmap = plt.cm.RdBu_r

    # --- Mann-Whitney U tests vs reference timepoint ---
    mw_results = {}
    ref_idx = None
    if ref_timepoint and ref_timepoint in timepoints:
        ref_idx = timepoints.index(ref_timepoint)
        ref_col = interaction_matrix[:, ref_idx]
        ref_valid = ref_col[~np.isnan(ref_col)]

        print(f"\n  Mann-Whitney U tests (reference: {ref_timepoint}):")
        for j, tp in enumerate(timepoints):
            if j == ref_idx:
                continue
            col = interaction_matrix[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) < 2 or len(ref_valid) < 2:
                mw_results[j] = (np.nan, 'n/a')
                continue
            try:
                stat, pval = mannwhitneyu(ref_valid, valid,
                                          alternative='two-sided')
                mw_results[j] = (pval, _pval_to_stars(pval))
                print(f"    {ref_timepoint} vs {tp}: U={stat:.1f}, "
                      f"p={pval:.4e} {_pval_to_stars(pval)}")
            except ValueError as e:
                print(f"    {ref_timepoint} vs {tp}: test failed ({e})")
                mw_results[j] = (np.nan, 'n/a')
    elif ref_timepoint:
        print(f"  Warning: reference timepoint '{ref_timepoint}' "
              f"not found in {timepoints}")

    # --- Decide which stats to plot and in which files ---
    if summary_stat == 'both':
        stat_sets = ['mean', 'median', 'both']
    else:
        stat_sets = [summary_stat]

    os.makedirs(output_dir, exist_ok=True)

    # Pre-compute stats (handle all-NaN columns gracefully)
    means   = np.full(len(timepoints), np.nan)
    medians = np.full(len(timepoints), np.nan)
    sems    = np.full(len(timepoints), np.nan)
    for j in range(len(timepoints)):
        col = interaction_matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            means[j]   = np.mean(valid)
            medians[j] = np.median(valid)
            sems[j]    = np.std(valid) / np.sqrt(len(valid))

    for stat_mode in stat_sets:
        fig, ax = plt.subplots(figsize=(3.5, 2.5))

        # -- Individual data points (jittered) --
        rng = np.random.default_rng(42)
        for j in range(len(timepoints)):
            col = interaction_matrix[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                continue
            pos = timepoint_positions[j]
            jitter = rng.uniform(-0.6, 0.6, size=len(valid))

            if color_dots:
                colors = dot_cmap(dot_norm(valid))
                ax.scatter(pos + jitter, valid, c=colors, s=12,
                           alpha=0.7, edgecolors='black', linewidths=0.3,
                           zorder=1)
            else:
                ax.scatter(pos + jitter, valid, color='black', s=12,
                           alpha=0.4, edgecolors='none', zorder=1)

        # -- Overlay stat markers (no connecting line) --
        if stat_mode in ('mean', 'both'):
            ax.errorbar(timepoint_positions, means, yerr=sems,
                        fmt='o', color='darkred', markersize=5,
                        linewidth=0, capsize=4, capthick=0.8,
                        elinewidth=0.8,
                        label='Mean +/- SEM', zorder=3)
        if stat_mode in ('median', 'both'):
            ax.errorbar(timepoint_positions, medians, yerr=sems,
                        fmt='s', color='#1a5276', markersize=4,
                        linewidth=0, capsize=3, capthick=0.8,
                        elinewidth=0.8,
                        label='Median +/- SEM', zorder=3)

        # -- Significance annotations --
        if mw_results and ref_idx is not None:
            # Get y-range for positioning brackets
            all_valid = interaction_matrix[~np.isnan(interaction_matrix)]
            if len(all_valid) > 0:
                data_max = np.max(all_valid)
            else:
                data_max = 1.0
            y_offset = data_max * 0.05
            bracket_y = data_max + y_offset

            for j in sorted(mw_results.keys()):
                pval, stars = mw_results[j]
                if stars == 'n/a' or stars == 'ns':
                    annotation = 'ns'
                else:
                    annotation = stars

                # Draw bracket between ref and comparison timepoint
                ref_x = timepoint_positions[ref_idx]
                cmp_x = timepoint_positions[j]

                ax.plot([ref_x, ref_x, cmp_x, cmp_x],
                        [bracket_y - y_offset * 0.3, bracket_y,
                         bracket_y, bracket_y - y_offset * 0.3],
                        color='black', linewidth=0.5, zorder=4)
                ax.text((ref_x + cmp_x) / 2, bracket_y + y_offset * 0.1,
                        annotation, ha='center', va='bottom',
                        fontsize=5, zorder=4)

                bracket_y += y_offset * 2.5  # stack brackets

        # -- Title reflecting which stat is shown --
        if stat_mode == 'both':
            title_stat = 'Mean & Median'
        else:
            title_stat = stat_mode.capitalize()
        ax.set_title(f'Hi-C Interaction Dynamics — {title_stat}')

        _style_summary_ax(ax, timepoint_positions, timepoint_labels,
                          ylabel, ymax)
        ax.legend(frameon=False)
        plt.tight_layout()

        # -- Save with stat in filename --
        suffix = f'_{stat_mode}' if summary_stat == 'both' else ''
        for ext in ('png', 'svg'):
            path = f'{output_dir}/{output_prefix}{suffix}.{ext}'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white',
                        format=ext)
            print(f"Saved: {path}")
        plt.close()


def create_interaction_heatmap(interaction_matrix, regions, timepoints,
                               normalize=True, oe_normalize=False,
                               output_prefix='interaction_heatmap',
                               output_dir='parascaris_triangle_plots',
                               vmin=None, vmax=None, cmap='YlOrRd',
                               cpm=False, log2_oe=False):
    """Seaborn heatmap of region x timepoint interaction scores."""

    n_regions    = len(regions)
    n_timepoints = len(timepoints)

    cell_height = 0.4
    cell_width  = 2.0
    fig_height  = cell_height * n_regions + 2
    fig_width   = cell_width * n_timepoints + 5

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    user_vmin = vmin is not None
    user_vmax = vmax is not None

    if vmin is None:
        vmin = np.nanmin(interaction_matrix)
    if vmax is None:
        vmax = np.nanmax(interaction_matrix)

    # Diverging colourmap for log2(O/E) — symmetric around 0
    # Only auto-symmetrize if user didn't specify scale limits
    if log2_oe and oe_normalize:
        cmap = 'RdBu_r'
        if not user_vmin and not user_vmax:
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max
        elif not user_vmin:
            vmin = -abs(vmax)
        elif not user_vmax:
            vmax = abs(vmin)

    timepoint_labels = [tp.replace('hr', '') if 'hr' in tp else tp
                        for tp in timepoints]

    # Colourbar label
    if log2_oe and oe_normalize:
        cbar_label = 'log2(O/E)'
    elif oe_normalize:
        cbar_label = 'O/E'
    elif cpm:
        cbar_label = 'Mean Contact Frequency (CPM)'
    elif normalize:
        cbar_label = 'Mean Contact Frequency (Normalized)'
    else:
        cbar_label = 'Mean Contact Frequency'

    sns.heatmap(interaction_matrix,
                xticklabels=timepoint_labels,
                yticklabels=[r['name'] for r in regions],
                cmap=cmap, vmin=vmin, vmax=vmax,
                cbar_kws={'label': cbar_label},
                linewidths=0.5, linecolor='gray',
                square=False, ax=ax)

    ax.set_xlabel('Developmental Timepoint (Hr)')
    ax.set_ylabel('Interacting Region Pairs')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, ha='right')

    # Title
    title = 'Hi-C Interaction Frequencies Across Development'
    if log2_oe and oe_normalize:
        title += '\n(log2(O/E))'
    elif oe_normalize:
        title += '\n(O/E Normalized)'
    elif cpm:
        title += '\n(CPM)'
    elif normalize:
        title += '\n(Normalized by Total Contacts)'
    plt.title(title)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    for ext in ('png', 'svg'):
        path = f'{output_dir}/{output_prefix}.{ext}'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white',
                    format=ext)
        print(f"Saved: {path}")
    plt.close()

    return interaction_matrix


def save_interaction_matrix_csv(interaction_matrix, regions, timepoints,
                                output_prefix, output_dir):
    """Save the region x timepoint score matrix as CSV."""
    df = pd.DataFrame(interaction_matrix,
                      index=[r['name'] for r in regions],
                      columns=timepoints)
    for i, region in enumerate(regions):
        df.loc[df.index[i], 'chr']     = region['chr']
        df.loc[df.index[i], 'region1'] = f"{region['start1']}-{region['end1']}"
        df.loc[df.index[i], 'region2'] = f"{region['start2']}-{region['end2']}"

    csv_path = f'{output_dir}/{output_prefix}.csv'
    df.to_csv(csv_path)
    print(f"Saved interaction matrix: {csv_path}")
    return df


# ===========================================================================
#  Configuration helpers
# ===========================================================================

def get_matrix_files(resolution):
    """Return dict of timepoint -> matrix file path."""
    res_str = f"{resolution // 1000}kb"
    return {
        '10hr':   f'matrix_files_{res_str}/prepde/pu_10hr_{res_str}_iced.matrix',
        '13.5hr': f'matrix_files_{res_str}/prepde/pu_13.5hr_{res_str}_iced.matrix',
        '17hr':   f'matrix_files_{res_str}/prepde/pu_17hr_{res_str}_iced.matrix',
        '24hr':   f'matrix_files_{res_str}/postpde/pu_24hr_{res_str}_iced.matrix',
        '36hr':   f'matrix_files_{res_str}/postpde/pu_36hr_{res_str}_iced.matrix',
        '48hr':   f'matrix_files_{res_str}/postpde/pu_48hr_{res_str}_iced.matrix',
        '60hr':   f'matrix_files_{res_str}/postpde/pu_60hr_{res_str}_iced.matrix',
        '72hr':   f'matrix_files_{res_str}/postpde/pu_72hr_{res_str}_iced.matrix',
    }


def get_timepoint_category(timepoint):
    """Classify timepoint as 'prepde' or 'postpde'."""
    if timepoint in ('10hr', '13.5hr', '17hr'):
        return 'prepde'
    elif timepoint in ('24hr', '36hr', '48hr', '60hr', '72hr'):
        return 'postpde'
    return None


# ===========================================================================
#  Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create interaction heatmap from Hi-C data with O/E normalization')
    parser.add_argument('--regions', type=str,
                        default='parascaris_interacting_regions.bed',
                        help='BED file with interacting region pairs')
    parser.add_argument('--timepoints', type=str,
                        default='10hr,17hr,24hr,48hr,72hr',
                        help='Comma-separated list of timepoints')
    parser.add_argument('--resolution', type=int, default=20000,
                        help='Hi-C matrix resolution in bp')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize by total contacts')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                        help='Do not normalize by total contacts')
    parser.add_argument('--cpm', action='store_true', default=False,
                        help='Normalize to counts per million (CPM)')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Minimum value for color scale')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Maximum value for color scale')
    parser.add_argument('--ymax', type=float, default=None,
                        help='Maximum y-value for summary plot')
    parser.add_argument('--cmap', type=str, default='YlOrRd',
                        help='Colormap for heatmap')
    parser.add_argument('--output', type=str, default='interaction_heatmap',
                        help='Output file prefix')
    parser.add_argument('--bin-bed', type=str, default=None,
                        help='Path to bin mapping bed for prepde samples')
    parser.add_argument('--soma-bin-bed', type=str, default=None,
                        help='Path to bin mapping bed for postpde/somatic samples')
    parser.add_argument('--germ-to-soma-mapping', type=str, default=None,
                        help='Path to germline-to-somatic chromosome mapping')
    parser.add_argument('--oe-normalize', action='store_true',
                        help='Use observed/expected normalization')
    parser.add_argument('--log2-oe', action='store_true',
                        help='Apply log2 to O/E values')
    parser.add_argument('--method', type=str, default='mean',
                        choices=['mean', 'max', 'sum', 'percentile95'],
                        help='Method for scoring interactions')
    parser.add_argument('--output-dir', type=str,
                        default='parascaris_interaction_heatmaps',
                        help='Output directory')
    parser.add_argument('--summary-stat', type=str, default='both',
                        choices=['mean', 'median', 'both'],
                        help='Stat to overlay on summary plot: mean, median, or both')
    parser.add_argument('--color-dots', action='store_true', default=False,
                        help='Color individual dots on summary plot using RdBu_r scale')
    parser.add_argument('--ref-timepoint', type=str, default=None,
                        help='Reference timepoint for Mann-Whitney U tests '
                             '(e.g. 17hr). Compared against all others.')

    args = parser.parse_args()

    # --- Auto-detect paths --------------------------------------------------
    if args.bin_bed is None:
        args.bin_bed = f"{args.resolution}/data1_{args.resolution}_abs.bed"
    if args.soma_bin_bed is None:
        args.soma_bin_bed = f"{args.resolution}/data1_{args.resolution}_abs_somatic.bed"
    if args.germ_to_soma_mapping is None:
        # Default path — update to match your local directory structure
        args.germ_to_soma_mapping = "germ_to_soma_mapping.bed"

    # --- Warn about CPM + O/E redundancy  (FIX #3) -------------------------
    if args.cpm and args.oe_normalize:
        print("\n" + "!" * 60)
        print("NOTE: --cpm is redundant with --oe-normalize.")
        print("CPM is a scalar multiplier that cancels in the O/E ratio.")
        print("O/E scores will be computed WITHOUT CPM scaling.")
        print("CPM will still be noted in labels for provenance.")
        print("!" * 60 + "\n")

    # --- Print configuration ------------------------------------------------
    print("=" * 60)
    print("Hi-C Interaction Heatmap Analysis")
    print("=" * 60)
    print(f"Regions file:          {args.regions}")
    print(f"Timepoints:            {args.timepoints}")
    print(f"Resolution:            {args.resolution} bp")
    print(f"Scoring method:        {args.method}")
    print(f"Sample normalization:  "
          f"{'CPM' if args.cpm else ('ON' if args.normalize else 'OFF')}")
    print(f"O/E normalization:     {'ON' if args.oe_normalize else 'OFF'}")
    if args.oe_normalize and args.log2_oe:
        print(f"Log2(O/E):             ON")
    if args.vmax is not None:
        print(f"Color scale max:       {args.vmax}")
    if args.ymax is not None:
        print(f"Summary plot max:      {args.ymax}")
    print(f"Prepde bin bed:        {args.bin_bed}")
    print(f"Postpde bin bed:       {args.soma_bin_bed}")
    print(f"Germ->soma mapping:    {args.germ_to_soma_mapping}")
    print("=" * 60)

    # --- Load germline -> somatic mapping -----------------------------------
    print(f"\nReading germline-to-somatic mapping...")
    germ_to_soma_mappings = read_germ_to_soma_mapping(args.germ_to_soma_mapping)

    # --- Load bin mappings (now returns chrom_bin_index too) -----------------
    bin_mappings = {}

    if os.path.exists(args.bin_bed):
        print(f"\nReading PREPDE bin mapping from {args.bin_bed}...")
        (bin_to_coords, coord_to_bin,
         chrom_info, chrom_bin_index) = read_bin_mapping(args.bin_bed,
                                                          args.resolution)
        bin_mappings['prepde'] = {
            'bin_to_coords':         bin_to_coords,
            'coord_to_bin':          coord_to_bin,
            'chrom_info':            chrom_info,
            'chrom_bin_index':       chrom_bin_index,
            'germ_to_soma_mappings': None,   # prepde uses germline coords
        }
        print(f"Loaded mapping for {len(bin_to_coords)} bins (prepde)")
    else:
        print(f"\nWarning: Prepde bin mapping not found: {args.bin_bed}")
        bin_mappings['prepde'] = None

    if os.path.exists(args.soma_bin_bed):
        print(f"\nReading POSTPDE bin mapping from {args.soma_bin_bed}...")
        (soma_bin_to_coords, soma_coord_to_bin,
         soma_chrom_info, soma_chrom_bin_index) = read_bin_mapping(
            args.soma_bin_bed, args.resolution)
        bin_mappings['postpde'] = {
            'bin_to_coords':         soma_bin_to_coords,
            'coord_to_bin':          soma_coord_to_bin,
            'chrom_info':            soma_chrom_info,
            'chrom_bin_index':       soma_chrom_bin_index,
            'germ_to_soma_mappings': germ_to_soma_mappings,
        }
        print(f"Loaded mapping for {len(soma_bin_to_coords)} bins (postpde)")
    else:
        print(f"\nWarning: Postpde bin mapping not found: {args.soma_bin_bed}")
        bin_mappings['postpde'] = None

    # --- Load regions -------------------------------------------------------
    if not os.path.exists(args.regions):
        print(f"Error: Regions file not found: {args.regions}")
        sys.exit(1)

    regions = read_interacting_regions(args.regions)
    print(f"\nLoaded {len(regions)} interacting region pairs")

    # Warn if windows are too small for the resolution
    if regions:
        sample = regions[0]
        w1 = sample['end1'] - sample['start1']
        w2 = sample['end2'] - sample['start2']
        min_window = min(w1, w2)
        if min_window < args.resolution:
            print(f"\n{'!' * 60}")
            print(f"WARNING: Region window size ({min_window} bp) is smaller "
                  f"than matrix resolution ({args.resolution} bp).")
            print(f"Each region maps to at most 1 bin, so interaction scores")
            print(f"depend on a single bin-pair. This is especially problematic")
            print(f"for post-PDE inter-chromosomal contacts which are sparse.")
            print(f"Consider using larger windows (>= 2x resolution = "
                  f"{2 * args.resolution} bp).")
            print(f"{'!' * 60}\n")

    region_chroms = set(r['chr'] for r in regions)
    print(f"Chromosomes in regions: {', '.join(sorted(region_chroms))}")

    print(f"\nFirst 3 regions:")
    for region in regions[:3]:
        print(f"  {region['name']}: {region['chr']}:"
              f"{region['start1']}-{region['end1']} x "
              f"{region['start2']}-{region['end2']}")
        dist = abs((region['start1'] + region['end1']) // 2 -
                   (region['start2'] + region['end2']) // 2)
        print(f"    Distance: {dist / 1e6:.2f} Mb")
        if germ_to_soma_mappings:
            sc1, sp1 = translate_germline_to_somatic(
                region['chr'], region['start1'], germ_to_soma_mappings)
            sc2, sp2 = translate_germline_to_somatic(
                region['chr'], region['start2'], germ_to_soma_mappings)
            if sc1 != region['chr'] or sc2 != region['chr']:
                print(f"    Somatic: {sc1}:{sp1} x {sc2}:{sp2}")

    if len(regions) > 3:
        print(f"\nLast 3 regions:")
        for region in regions[-3:]:
            print(f"  {region['name']}: {region['chr']}:"
                  f"{region['start1']}-{region['end1']} x "
                  f"{region['start2']}-{region['end2']}")
            if germ_to_soma_mappings:
                sc1, _ = translate_germline_to_somatic(
                    region['chr'], region['start1'], germ_to_soma_mappings)
                sc2, _ = translate_germline_to_somatic(
                    region['chr'], region['start2'], germ_to_soma_mappings)
                if sc1 != region['chr'] or sc2 != region['chr']:
                    print(f"    Somatic: {sc1} x {sc2}")

    # --- Load Hi-C matrices -------------------------------------------------
    timepoints   = [t.strip() for t in args.timepoints.split(',')]
    matrix_files = get_matrix_files(args.resolution)

    print(f"\nLoading Hi-C matrices for {len(timepoints)} timepoints...")
    hic_matrices   = {}
    total_contacts = {}

    for tp in timepoints:
        if tp not in matrix_files:
            print(f"Warning: No matrix file for timepoint {tp}")
            continue
        matrix, contacts = read_hic_matrix(matrix_files[tp], args.resolution)
        hic_matrices[tp]   = matrix
        total_contacts[tp] = contacts

    # --- CPM / median normalization factors ---------------------------------
    if args.cpm:
        norm_factors = {tp: 1e6 / tc if tc > 0 else 1.0
                        for tp, tc in total_contacts.items()}
        print(f"\nCPM normalization factors:")
        for tp, nf in norm_factors.items():
            print(f"  {tp}: {nf:.6f}  (total contacts: {total_contacts[tp]:.0f})")
    elif args.normalize:
        median_contacts = np.median(list(total_contacts.values()))
        norm_factors = {tp: median_contacts / tc if tc > 0 else 1.0
                        for tp, tc in total_contacts.items()}
        print(f"\nMedian total contacts: {median_contacts:.0f}")
    else:
        norm_factors = {tp: 1.0 for tp in timepoints}

    # --- Chromosome-aware expected values for O/E ---------------------------
    # KEY INSIGHT: All Hi-C matrices are in GERMLINE coordinate space.
    # Scoring always uses germline bin mapping.
    # For post-PDE expected: group germline bins by somatic chromosome.
    germ_bm = bin_mappings.get('prepde')
    if germ_bm is None:
        print("ERROR: Germline (prepde) bin mapping is required.")
        sys.exit(1)

    germ_coord_to_bin   = germ_bm['coord_to_bin']
    germ_chrom_bin_index = germ_bm['chrom_bin_index']
    germ_bin_to_coords  = germ_bm['bin_to_coords']

    expected_data = {}   # tp -> (intra_expected, inter_expected)
    if args.oe_normalize:
        print(f"\nCalculating chromosome-aware expected values...")

        # Build somatic chromosome grouping of germline bins (for post-PDE)
        soma_chrom_bin_index = None
        if germ_to_soma_mappings:
            print(f"\n  Building somatic chromosome index from germline bins...")
            soma_chrom_bin_index = build_somatic_chrom_bin_index(
                germ_coord_to_bin, germ_to_soma_mappings)

        for tp in timepoints:
            if tp not in hic_matrices or hic_matrices[tp] is None:
                continue

            tp_cat = get_timepoint_category(tp)
            print(f"\n  === {tp} ({tp_cat}) ===")

            if tp_cat == 'postpde' and soma_chrom_bin_index:
                # Post-PDE: use somatic chromosome grouping of germline bins
                intra_exp, inter_exp = calculate_expected_chrom_aware(
                    hic_matrices[tp],
                    germ_bin_to_coords,
                    soma_chrom_bin_index,
                    args.resolution)
            else:
                # Pre-PDE: use germline chromosome structure
                intra_exp, inter_exp = calculate_expected_chrom_aware(
                    hic_matrices[tp],
                    germ_bin_to_coords,
                    germ_chrom_bin_index,
                    args.resolution)
            expected_data[tp] = (intra_exp, inter_exp)

    # --- Score all region x timepoint pairs ---------------------------------
    # ALL scoring uses germline bin mapping (matrices are germline-indexed).
    # germ_to_soma_mappings is only used in O/E to determine same_chrom.
    print(f"\nCalculating interaction scores...")
    interaction_matrix = np.zeros((len(regions), len(timepoints)))

    for i, region in enumerate(regions):
        if i == 0 or i % 10 == 0:
            print(f"  Processing region {i+1}/{len(regions)}: {region['name']}")

        for j, tp in enumerate(timepoints):
            if tp not in hic_matrices or hic_matrices[tp] is None:
                interaction_matrix[i, j] = np.nan
                continue

            tp_cat = get_timepoint_category(tp)

            # For O/E: postpde needs germ_to_soma to determine same_chrom
            # For scoring: always use germline bins (no coordinate translation)
            if args.oe_normalize and tp in expected_data:
                intra_exp, inter_exp = expected_data[tp]
                oe_germ_to_soma = (germ_to_soma_mappings
                                   if tp_cat == 'postpde' else None)
                score = calculate_oe_interaction_score(
                    hic_matrices[tp], region,
                    germ_coord_to_bin, args.resolution,
                    args.method,
                    None,              # no translation for bin lookup
                    intra_exp, inter_exp,
                    args.log2_oe,
                    oe_germ_to_soma)   # only for same_chrom determination
                interaction_matrix[i, j] = score
            else:
                score = calculate_interaction_score(
                    hic_matrices[tp], region,
                    germ_coord_to_bin, args.resolution,
                    args.method,
                    None)              # no translation for bin lookup
                if args.normalize or args.cpm:
                    interaction_matrix[i, j] = score * norm_factors.get(tp, 1.0)
                else:
                    interaction_matrix[i, j] = score

    # --- Summary statistics -------------------------------------------------
    print(f"\nInteraction matrix statistics:")
    print(f"  Min:    {np.nanmin(interaction_matrix):.4f}")
    print(f"  Max:    {np.nanmax(interaction_matrix):.4f}")
    print(f"  Mean:   {np.nanmean(interaction_matrix):.4f}")
    print(f"  Median: {np.nanmedian(interaction_matrix):.4f}")
    print(f"  95th %%: {np.nanpercentile(interaction_matrix, 95):.4f}")

    # Flag problematic regions
    print(f"\nChecking for problematic regions:")
    all_zero = [r['name'] for i, r in enumerate(regions)
                if np.all(interaction_matrix[i, :] == 0)]
    all_nan  = [r['name'] for i, r in enumerate(regions)
                if np.all(np.isnan(interaction_matrix[i, :]))]

    if all_zero:
        print(f"  Regions with all zeros ({len(all_zero)}):")
        for name in all_zero[:5]:
            print(f"    - {name}")
        if len(all_zero) > 5:
            print(f"    ... and {len(all_zero) - 5} more")
    if all_nan:
        print(f"  Regions with all NaN ({len(all_nan)}):")
        for name in all_nan[:5]:
            print(f"    - {name}")
        if len(all_nan) > 5:
            print(f"    ... and {len(all_nan) - 5} more")
    if not all_zero and not all_nan:
        print("  None found.")

    # --- Generate outputs ---------------------------------------------------
    print(f"\nCreating interaction heatmap...")
    create_interaction_heatmap(
        interaction_matrix, regions, timepoints,
        normalize=args.normalize, oe_normalize=args.oe_normalize,
        output_prefix=args.output, output_dir=args.output_dir,
        vmin=args.vmin, vmax=args.vmax, cmap=args.cmap,
        cpm=args.cpm, log2_oe=args.log2_oe)

    save_interaction_matrix_csv(interaction_matrix, regions, timepoints,
                                args.output, args.output_dir)

    print(f"\nCreating summary plot(s)...")
    create_summary_plot(
        interaction_matrix, regions, timepoints,
        output_prefix=f"{args.output}_summary",
        output_dir=args.output_dir,
        ymax=args.ymax,
        oe_normalize=args.oe_normalize,
        log2_oe=args.log2_oe,
        summary_stat=args.summary_stat,
        color_dots=args.color_dots,
        dot_vmin=args.vmin,
        dot_vmax=args.vmax,
        ref_timepoint=args.ref_timepoint)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Output files in {args.output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()