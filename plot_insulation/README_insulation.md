# insulation_analysis

Plot Hi-C insulation scores across *Ascaris* developmental stages for each
chromosome.  Visualizes boundary strength dynamics before and after programmed
DNA elimination (PDE), with eliminated DNA regions highlighted.

## Overview

`plot_insulation.py` generates per-chromosome stacked insulation score plots
(positive = strong boundaries in blue, negative = weak boundaries in red),
a comprehensive box plot comparing insulation in end vs. internal vs. retained
regions, and a statistical summary.

Key features:
- **Pre-PDE stages** show continuous insulation profiles including eliminated
  regions.
- **Post-PDE stages** have data masked in eliminated regions with trace breaks
  at elimination boundaries.
- **Floor value removal**: artificially low scores (FAN-C artifacts at
  low-mappability bins) are detected per region type and removed before
  analysis.
- **Box plot** compares insulation in three region categories (chromosome ends,
  internal breaks, retained) across all stages, with Mann-Whitney U
  significance testing.

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
| `data/AG_v50_eliminated_strict.bed` | Yes | Eliminated regions BED with type column (shared) |
| Insulation bedgraphs | No | FAN-C insulation scores in `insulation_scores/{prepde,postpde_mapped}/` |

Insulation scores are computed by [FAN-C](https://fan-c.readthedocs.io/)
(`fanc insulation`) from Hi-C matrices.  Raw reads are available from GEO [GSE314626](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE314626).

## Usage

```bash
python plot_insulation.py \
    --prepde-dir insulation_scores/prepde \
    --postpde-dir insulation_scores/postpde_mapped \
    --eliminated-bed data/AG_v50_eliminated_strict.bed \
    --prepde-stages teste ovary 0hr 48hr 60hr \
    --postpde-stages 5day 10day \
    --y-min -2 --y-max 1 \
    --boxplot-y-min -5 --boxplot-y-max 3 \
    --floor-threshold -10 \
    --floor-frequency 0.01 \
    --output-dir results/
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--prepde-dir` | `insulation_scores/prepde` | Directory with pre-PDE insulation files |
| `--postpde-dir` | `insulation_scores/postpde_mapped` | Directory with post-PDE mapped files |
| `--eliminated-bed` | `AG.v50.eliminated_strict.bed` | BED file with eliminated regions (4 cols: chr start end type) |
| `--prepde-stages` | `teste ovary ... 60hr` | Pre-PDE stages in order |
| `--postpde-stages` | `4day 5day 10day` | Post-PDE stages in order |
| `--y-min` / `--y-max` | -4 / 1 | Y-axis limits for chromosome plots |
| `--boxplot-y-min/max` | -10 / 2 | Y-axis limits for box plot |
| `--floor-threshold` | -10 | Values below this are floor candidates |
| `--floor-frequency` | 0.01 | Frequency threshold for floor detection (0.01 = 1%) |
| `--chromosomes` | auto | Specific chromosomes to analyze |

## Outputs

Per chromosome:
- `insulation_{chr}.png` / `.pdf` / `.svg` — stacked insulation plot

Summary:
- `comprehensive_boxplot.png` / `.pdf` / `.svg` — box plot by region type
- `statistical_summary.csv` — Mann-Whitney U test results

## Figure mapping

| Figure | Stages | Y-limits | Notes |
|--------|--------|----------|-------|
| Fig. 3C, 3D | teste through 10day | -2 to 1 | Floor values removed |
