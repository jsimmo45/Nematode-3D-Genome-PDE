# saddle_plot_analysis

Compartment saddle plots and comprehensive compartmentalization analysis for
*Ascaris* Hi-C data across developmental timepoints.

## Overview

Two scripts are provided:

- **`saddle_plots.py`** — Core saddle plot generation and compartment strength
  quantification.  Produces per-timepoint saddle plots and the
  (AA+BB)/(AB+BA) strength metric across development.

- **`saddle_plots_full.py`** — Extended analysis that includes everything in
  `saddle_plots.py` plus additional analyses:
  1. Saddle plots and strength quantification
  2. Sequential compartment switching between consecutive stages
  3. CBR overlap analysis
  4. Compartment boundary sharpness (PC1 gradient)
  5. Entropy analysis of compartmentalization
  6. Compartment strength (mean |PC1|) and A/B separation

Use `saddle_plots.py` to reproduce the core saddle plot figures.  Use
`saddle_plots_full.py` for the full suite of compartment analyses.

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
| Eigenvector files | Yes | In `data/eigenvectors/ascaris/{prepde,postpde}/` |
| HiC-Pro sparse matrices | No | 100kb ICE-normalized; regenerate from raw reads |
| `data/cbr_v50_500kb_windows_labeled.bed` | Yes | CBR positions (optional, for `saddle_plots_full.py`) |

Raw reads: GEO [GSE314626](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE314626).

## Usage

### Core saddle plots

```bash
python saddle_plots.py \
    --matrix-dir matrix_files_100kb \
    --eigenvector-dir data/eigenvectors/ascaris \
    --timepoints 0hr 48hr 60hr 5day 10day \
    --pre-pde-stages 0hr 48hr 60hr \
    --post-pde-stages 5day 10day \
    --output-dir results/
```

### Full compartment analysis

```bash
python saddle_plots_full.py \
    --matrix-dir matrix_files_100kb \
    --eigenvector-dir data/eigenvectors/ascaris \
    --cbr-bed data/cbr_v50_500kb_windows_labeled.bed \
    --timepoints teste ovary 0hr 48hr 60hr 5day 10day \
    --pre-pde-stages teste ovary 0hr 48hr 60hr \
    --post-pde-stages 5day 10day \
    --output-dir results/
```

## Parameters (shared by both scripts)

| Flag | Default | Description |
|------|---------|-------------|
| `--matrix-dir` | `matrix_files_100kb` | Base directory for matrices (prepde/postpde subdirs) |
| `--eigenvector-dir` | `.` | Base directory for eigenvectors |
| `--timepoints` | `teste...10day` | All timepoints in order |
| `--pre-pde-stages` | `teste...60hr` | Pre-PDE timepoints |
| `--post-pde-stages` | `5day 10day` | Post-PDE timepoints |
| `--matrix-pattern` | `as_{}_iced_100kb.matrix` | Matrix filename pattern |
| `--eigen-pattern` | `as_{}_iced_100kb.matrix.eigenvector` | Eigenvector filename pattern |
| `--max-bins` | 5000 | Limit bins for faster processing |
| `--pc1-threshold` | 0.005 | Min \|PC1\| for compartment calls |
| `--output-dir` | `output` | Output directory |

Additional flags for `saddle_plots_full.py`:

| Flag | Default | Description |
|------|---------|-------------|
| `--cbr-bed` | none | CBR BED file for boundary overlap analysis |
| `--chromosome-mapping` | none | Pre-to-post PDE chromosome mapping |

## Outputs

### From `saddle_plots.py`

- `saddle_plot_{timepoint}.png/.svg` — per-stage saddle plots
- `saddle_strength_across_development.png/.svg` — strength trend line
- `saddle_strength_summary.csv`

### Additional from `saddle_plots_full.py`

- `saddle_strength_decomposition.png/.svg` — AA, BB, AB components
- `switching_*.png/.svg` — compartment switching between stages
- `boundary_*.png` — boundary sharpness analysis
- `entropy_*.png` — entropy analysis
- `compartment_strength_*.png` — mean |PC1| and A/B separation

## Figure mapping

| Figure | Script | Notes |
|--------|--------|-------|
| Fig. 5C | `saddle_plots.py` | Saddle plots, 20 quantiles, all stages |
| Fig. S8A, S8B, S8C | `saddle_plots_full.py` | Strength decomposition, switching |
