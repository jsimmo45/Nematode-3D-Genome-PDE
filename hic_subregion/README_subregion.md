# hic_subregion

Plot Hi-C contact heatmaps for arbitrary genomic regions in *Ascaris*.
Extracts a user-defined subregion from a HiC-Pro sparse matrix and visualizes
it as a CPM-normalized contact map.

## Overview

`plot_hic_subregion.py` takes any pair of genomic coordinate ranges and
produces a rectangular (or square) Hi-C contact heatmap.  Useful for
visualizing specific intra- or inter-chromosomal interactions at any
resolution, such as contacts between CBR-flanking regions on different
chromosomes.

Features:
- Flexible coordinate specification: single or multi-chromosome regions per axis
- CPM normalization (optional)
- Custom white→yellow→red→black colormap
- Chromosome boundary lines for multi-chromosome views
- Configurable tick intervals and color scale

## Dependencies

```
numpy
pandas
matplotlib
```

## Input files

| File | In repo? | Description |
|------|----------|-------------|
| HiC-Pro bins BED | Yes (shared) | `AG_10kb_hicpro_bins.bed` etc. in `data/` |
| ICE-normalized matrices | No | HiC-Pro sparse matrices; regenerate from GEO [GSE314626](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE314626) |

## Usage

### Inter-chromosomal subregion

```bash
python plot_hic_subregion.py \
    --bins data/AG_10kb_hicpro_bins.bed \
    --matrix matrix/10kb/5day_10kb_iced.matrix \
    --x-coords 'chr01:7000000-12000000' \
    --y-coords 'chr06:2500000-7500000' \
    --vmax p99 \
    --tick-interval 500000 \
    --dpi 300 \
    --output results/chr01_vs_chr06
```

### Whole chromosome

```bash
python plot_hic_subregion.py \
    --bins data/AG_10kb_hicpro_bins.bed \
    --matrix matrix/10kb/5day_10kb_iced.matrix \
    --x-coords 'chr01:1-18470381' \
    --y-coords 'chr01:1-18470381' \
    --vmax 2 \
    --tick-interval 2000000 \
    --dpi 300 \
    --output results/chr01_full
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--bins` | required | HiC-Pro bins BED file (chr start end bin_id) |
| `--matrix` | required | HiC-Pro ICE-normalized sparse matrix |
| `--x-coords` | required | X-axis coordinates (`chr:start-end`, comma-separated for multi-chr) |
| `--y-coords` | required | Y-axis coordinates (same format) |
| `--output` | required | Output file prefix |
| `--vmax` | 20000 | Color scale max: numeric, `auto`, or `pNN` (percentile) |
| `--tick-interval` | 100000 | Coordinate tick spacing in bp |
| `--dpi` | 300 | PNG resolution |
| `--no-cpm` | off | Skip CPM normalization |

## Outputs

- `{prefix}.png` — raster heatmap
- `{prefix}.pdf` — vector heatmap

## Figure mapping

| Figure | Coords | vmax | Notes |
|--------|--------|------|-------|
| Fig. 1D | chr01 vs chr06, 5Mb windows | p99 | 10kb resolution, 2-4 cell |
