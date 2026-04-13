# eigenvector_genome

Plot eigenvector 1 (EV1 / PC1 compartment signal) across all *Ascaris*
chromosomes for multiple developmental stages in a stacked grid layout.

## Overview

`plot_eigenvector_genome.py` generates a large multi-panel figure showing
EV1 traces for every chromosome (columns) at every developmental stage (rows).
Autosomal traces are blue, sex chromosomes are magenta.  Post-PDE stages have
coordinates mapped back to germline space with gaps at eliminated regions.

Includes per-sample-chromosome sign correction (`--flip-combos`) since
eigenvector sign is arbitrary and must be manually corrected for biological
consistency across stages.

## Dependencies

```
numpy
matplotlib
scipy
```

## Input files

| File | In repo? | Description |
|------|----------|-------------|
| `data/AG_v50_eliminated_strict.bed` | Yes | Eliminated regions (shared) |
| `data/as_v50_prepde_to_postpde_coordinates.bed` | Yes | Coordinate mapping (shared) |
| `data/AG_v50_chrom_sizes.txt` | Yes | Chromosome sizes (shared) |
| Eigenvector files | Yes | FAN-C eigenvectors in `data/eigenvectors/ascaris/{prepde,postpde}/` |

## Usage

```bash
python plot_eigenvector_genome.py \
    --elimination-bed data/AG_v50_eliminated_strict.bed \
    --mapping-bed data/as_v50_prepde_to_postpde_coordinates.bed \
    --chrom-sizes data/AG_v50_chrom_sizes.txt \
    --pre-dir data/eigenvectors/ascaris/prepde \
    --post-dir data/eigenvectors/ascaris/postpde \
    --samples as_0hr_iced_100kb as_48hr_iced_100kb as_60hr_iced_100kb \
              as_5day_iced_100kb as_10day_iced_100kb \
    --display-names 'as_0hr_iced_100kb:1cell;as_48hr_iced_100kb:2-4cell;as_60hr_iced_100kb:4-8cell;as_5day_iced_100kb:32-64cell;as_10day_iced_100kb:L1' \
    --flip-combos '48hr,chr01;60hr,chr01;0hr,chr03;5day,chr03' \
    --smoothing 3 \
    --output-dir results/
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--elimination-bed` | required | Eliminated regions BED |
| `--mapping-bed` | required | Pre-to-post PDE coordinate mapping |
| `--chrom-sizes` | required | Chromosome sizes file |
| `--pre-dir` | required | Directory with pre-PDE eigenvector files |
| `--post-dir` | required | Directory with post-PDE eigenvector files |
| `--samples` | required | Sample filenames (e.g., `as_0hr_iced_100kb`) |
| `--display-names` | none | Display label mapping (`name:label;...`) |
| `--flip-combos` | none | EV1 sign corrections (`sample,chr;sample,chr;...`) |
| `--smoothing` | 0 | Moving average window in bins (0 = off) |
| `--output-dir` | `ev1_stacked_output` | Output directory |

## Outputs

- `eigen_stacked_{samples}_{smooth}.png` / `.svg` — genome-wide stacked EV1 plot

## Figure mapping

| Figure | Samples | Smoothing | Notes |
|--------|---------|-----------|-------|
| Fig. S7 | 1cell through L1 | 3-bin | y: -0.3 to 0.3, sign-corrected |
