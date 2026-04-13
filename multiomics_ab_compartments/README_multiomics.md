# multiomics_compartments

Compare multi-omics signal levels between A and B chromatin compartments in
*Ascaris suum* at the 5-day post-PDE timepoint.  For each of 8 omics datasets, genomic bins are classified as A or B
based on Hi-C eigenvector 1 (EV1) sign, and signal distributions are compared
with a Mann-Whitney U test.

## Overview

`multiomics_ab_compartments.py` takes an eigenvector file (100 kb bins) and a
collection of omics signal tracks (10 kb bedgraph files), averages the 10 kb
signal within each 100 kb eigenvector bin, assigns A/B compartment identity
based on EV1 sign, and produces per-dataset box-and-whisker plots with jittered
data points showing the signal distribution in A vs B compartments.

Currently configured for 8 datasets at the 5-day post-PDE timepoint: RNA-seq,
ATAC-seq, Pol II, and 5 histone modifications (H3K4me3, H3S10p, H3K36me2,
H3K36me3, H3K9me3).  Dataset definitions and signal filters are configured in
dictionaries at the top of the script.

## Dependencies

```
numpy
pandas
matplotlib
scipy
```

## Input files

| File | Description |
|------|-------------|
| `--eigenvector` | EV1 file at 100 kb resolution (FANC format: `chr  start  end  PC1`) |
| Omics bedgraphs | 10 kb signal tracks (`chr  start  end  value`) in subdirectories under `--base-dir` |
| `pre_to_post.bed` (optional) | Post-PDE → pre-PDE coordinate mapping, auto-detected in `--base-dir` |

The eigenvector file is included in the repository under
`data/eigenvectors/ascaris/`.  Binned omics signal tracks (`.mean0.10kb.bg`
files) are included in this repository under `omics_data/`.  Raw sequencing
reads are available from GEO [GSE314626](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE314626).

Expected directory layout under `--base-dir`:

```
omics_data/
├── rnaseq/
│   └── rna_seq_5day.norm.mean0.10kb.bg
├── atacseq/
│   └── atac_5day_*.mean0.10kb.bg
└── histones/
    ├── h3k4me3/
    │   └── H3K4me3_5day_*.mean0.10kb.bg
    ├── h3s10p/
    │   └── H3S10p_5day_*.mean0.10kb.bg
    ├── h3k36me2/
    │   └── H3K36me2_5day_*.mean0.10kb.bg
    ├── h3k36me3/
    │   └── H3K36me3_5day_*.mean0.10kb.bg
    ├── h3k9me3/
    │   └── H3K9me3_5day_*.mean0.10kb.bg
    └── pol2/
        └── Pol2_5day_*.mean0.10kb.bg
```

## Usage

```bash
python multiomics_ab_compartments.py \
    --eigenvector data/eigenvectors/ascaris/postpde/as_5day_iced_100kb.matrix.eigenvector \
    --base-dir omics_data/ \
    --ev-threshold 0.15 \
    --output-dir results/
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--eigenvector` | required | Eigenvector file (FANC format, 100 kb) |
| `--base-dir` | `.` | Base directory with omics subdirectories |
| `--ev-threshold` | 0.15 | Minimum \|EV1\| to include a bin (filters ambiguous compartment calls) |
| `--output-dir` | `multiomics_AB_results` | Output directory |

## In-script configuration

**`DATASETS`** — dict defining each omics dataset: display name, plot color,
bedgraph file glob pattern, subdirectory under `--base-dir`, and y-axis max.
Edit to add/remove datasets or change file patterns for different timepoints.

**`SIGNAL_FILTERS`** — dict of per-dataset minimum signal thresholds.  Bins
with signal below the threshold are excluded.  Default is 0.0 for all datasets
(no filtering).

## Outputs

Per dataset:
- `{dataset}_AB_comparison.png` / `.svg` — box plot with jittered points

Summary:
- `AB_compartment_statistics.csv` — n, mean, median, log2(A/B), p-value per dataset

## Figure mapping

| Figure | Eigenvector | Threshold | Notes |
|--------|-----------|-----------|-------|
| Fig. 5B | `as_5day_iced_100kb` | 0.15 | 8 datasets, no signal filter |
