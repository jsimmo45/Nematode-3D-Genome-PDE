# chromosome_ends

Analyze multi-omics signal at new chromosome ends created by programmed DNA
elimination (PDE) vs. internal break regions across *Ascaris* developmental
timepoints.

## Overview

`analyze_ends_timepoints.py` compares insulation scores, gene expression, and
histone modification levels between two classes of PDE break sites — terminal
(end) and internal (middle) — across 5 developmental stages spanning DNA
elimination.  This tests whether chromatin organization at new chromosome ends
differs from internal break sites during and after PDE.

For each window size around break sites, the script:
1. Loads a BED file defining end vs. middle break regions
2. Intersects each region with omics signal tracks (insulation, RNA-seq,
   ATAC-seq, H3K9me3, H3K4me3)
3. Produces dot+errorbar plots comparing end vs. middle at each timepoint
4. Runs Mann-Whitney U tests between consecutive stages and 1-cell vs. L1

## Dependencies

```
numpy
pandas
matplotlib
seaborn
scipy
```

## Input files

| File | In repo? | Description |
|------|----------|-------------|
| `data/new_ends_post_pde/AG.v50.new_ends_post_pde.{size}.bed` | Yes | Break region BED files at each window size (end/middle type in col 5) |
| Insulation bedgraphs | No | FAN-C insulation scores in `insulation_scores/{prepde,postpde_mapped}/` |
| RNA-seq bedgraphs | No | 50 kb binned expression in `rnaseq/50kb_binning/` |
| ATAC-seq bedgraphs | No | 50 kb binned accessibility in `atac/50kb_binning/` |
| H3K9me3 bedgraphs | No | 50 kb binned signal in `h3k9me3/50kb_binning/` |
| H3K4me3 bedgraphs | No | 50 kb binned signal in `h3k4me3/50kb_binning/` |

The break-region BED files are included in the repository.  Omics bedgraph
files are binned signal tracks generated from sequencing alignments.  Raw reads
are available from SRA under accession SRPXXXXXX.

## Usage

```bash
python analyze_ends_timepoints.py \
    --ends-dir data/new_ends_post_pde \
    --window-sizes '100kb,250kb,500kb,1000kb' \
    --plot-style points \
    --error-bar-type sem \
    --ylim-insulation '-4,2' \
    --ylim-rnaseq '0,500' \
    --ylim-atac '0,30' \
    --ylim-h3k9me3 '0,300' \
    --ylim-h3k4me3 '0,400' \
    --output-dir results/
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--ends-dir` | `data/new_ends_post_pde` | Directory with break-region BED files |
| `--window-sizes` | `100kb,250kb,500kb,1000kb` | Comma-separated window sizes to analyze |
| `--plot-style` | `points` | `points` (dot+errorbar) or `violin` |
| `--error-bar-type` | `sem` | `sem` or `std` |
| `--ylim-insulation` | auto | Y-axis limits (format: `min,max` or `auto`) |
| `--ylim-rnaseq` | auto | Y-axis limits for RNA-seq |
| `--ylim-atac` | auto | Y-axis limits for ATAC-seq |
| `--ylim-h3k9me3` | auto | Y-axis limits for H3K9me3 |
| `--ylim-h3k4me3` | auto | Y-axis limits for H3K4me3 |
| `--output-dir` | `plots` | Output directory |

## Outputs

Per window size:
- `chromosome_ends_timepoints_{size}.png` / `.svg` — multi-panel figure
- `chromosome_ends_timepoint_stats.txt` — statistical summary

## Figure mapping

| Figure | Window sizes | Notes |
|--------|-------------|-------|
| Fig. XI | 100kb, 250kb, 500kb, 1000kb | Points + SEM, 5 stages |
