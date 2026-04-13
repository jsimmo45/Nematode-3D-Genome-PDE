# insulation_genome

Plot insulation scores across all *Ascaris* chromosomes for multiple
developmental stages in a stacked grid layout.

## Overview

`plot_insulation_genome.py` generates a large multi-panel figure showing
insulation score traces for every chromosome (columns) at every developmental
stage (rows).  Autosomal traces are blue, sex chromosome traces are magenta.
Post-PDE stages have their coordinates mapped back to germline space, with
gaps at eliminated regions (shaded red).

Optional moving-average smoothing reduces noise while preserving boundary
features.  Post-PDE segment labels (somatic chromosome names) are annotated
above multi-segment chromosomes.

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
| Insulation bedgraphs | No | FAN-C insulation scores in prepde/ and postpde/ directories |

Insulation scores are computed by FAN-C (`fanc insulation`) from Hi-C
matrices.  Raw reads on GEO [GSE314626](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE314626).

## Usage

```bash
python plot_insulation_genome.py \
    --elimination-bed data/AG_v50_eliminated_strict.bed \
    --mapping-bed data/as_v50_prepde_to_postpde_coordinates.bed \
    --chrom-sizes data/AG_v50_chrom_sizes.txt \
    --pre-dir insulation_scores/prepde \
    --post-dir insulation_scores/postpde \
    --samples 0hr 48hr 60hr 5day 10day \
    --display-names '0hr:1cell;48hr:2-4cell;60hr:4-8cell;5day:32-64cell;10day:L1' \
    --smoothing 20 \
    --output-dir results/
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--elimination-bed` | required | Eliminated regions BED |
| `--mapping-bed` | required | Pre-to-post PDE coordinate mapping |
| `--chrom-sizes` | required | Chromosome sizes file |
| `--pre-dir` | required | Directory with pre-PDE insulation bedGraphs |
| `--post-dir` | required | Directory with post-PDE insulation bedGraphs |
| `--samples` | required | Sample names (e.g., `0hr 48hr 60hr 5day 10day`) |
| `--display-names` | none | Display label mapping (`name:label;name:label`) |
| `--smoothing` | 0 | Moving average window in bins (0 = off) |
| `--output-dir` | `insulation_plots` | Output directory |

## Outputs

- `insulation_stacked_{samples}_{smooth}.png` / `.svg` — genome-wide stacked plot

## Figure mapping

| Figure | Samples | Smoothing | Notes |
|--------|---------|-----------|-------|
| Fig. 3C,D | 1cell, 2-4cell, 4-8cell, 32-64cell, L1 | 20-bin | All chromosomes, y: -2 to 1 |
