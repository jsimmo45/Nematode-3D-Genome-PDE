# saddle_plots

Compartment saddle plot analysis for *Ascaris suum* Hi-C data across
developmental timepoints spanning programmed DNA elimination (PDE).

## Overview

`saddle_plots.py` computes saddle plots — contact enrichment matrices stratified
by eigenvector 1 (PC1) quantile — for each developmental stage, then quantifies
compartmentalization strength as the ratio of within-compartment to
between-compartment contact enrichment: (AA + BB) / (AB + BA).

The full analysis pipeline includes:

1. **Saddle plots** with optional global color scaling across timepoints
2. **Saddle strength** metrics decomposed into AA, BB, and AB components
3. **Compartment switching** between consecutive timepoints (A→B and B→A)
4. **Boundary sharpness** (mean |∇PC1| at compartment boundaries)
5. **Compartment entropy** (A/B transition frequency per bin)
6. **Compartment strength** (mean |PC1|) and A/B separation

All analyses can optionally compare chromosome break regions (CBR) against the
rest of the genome when a CBR BED file is provided.

## Dependencies

```
numpy
pandas
matplotlib
seaborn
scipy
```

## Input files

| File | Description |
|------|-------------|
| Hi-C matrices | ICE-normalized sparse matrices (HiC-Pro format) in `--matrix-dir/{prepde,postpde}/` |
| Eigenvector files | FANC format (`chr  start  end  PC1`) in `--eigenvector-dir/{prepde,postpde}/eigenvectors/` |
| `--cbr-bed` (optional) | BED file of chromosome break regions for CBR-vs-genome comparisons |
| `--chromosome-mapping` (optional) | BED mapping pre-PDE → post-PDE coordinates |

Hi-C matrices used by this script are ICE-normalized sparse matrices generated
by [HiC-Pro](https://github.com/nservant/HiC-Pro) at 100 kb resolution.  Raw
sequencing reads are available from SRA under accession SRPXXXXXX.  To
regenerate matrices, run HiC-Pro with the Ascaris v50 reference genome, then
apply ICE balancing.  Eigenvector files are included in this repository under
`data/eigenvectors/`.

## Usage

### Full analysis (saddle plots + switching + boundary + entropy)

```bash
python saddle_plots.py \
    --timepoints teste,ovary,0hr,48hr,60hr,5day,10day \
    --pre-pde-timepoints teste,ovary,0hr,48hr,60hr \
    --post-pde-timepoints 5day,10day \
    --matrix-dir matrix_files_100kb \
    --eigenvector-dir .. \
    --matrix-pattern 'as_{}_iced_100kb.matrix' \
    --eigenvector-pattern 'as_{}_iced_100kb.matrix.eigenvector' \
    --max-bins 5000 \
    --pc1-threshold 0.005 \
    --global-scale \
    --boundary-linewidth 4.0 \
    --boundary-color yellow \
    --cbr-bed data/cbr_v50_500kb_windows_labeled.bed \
    --stage-names 'teste:teste,ovary:ovary,0hr:1 cell,48hr:2-4 cell,60hr:4-8 cell,5day:32-64 cell,10day:L1' \
    --output-dir output/
```

### Saddle plots only (skip downstream analyses)

```bash
python saddle_plots.py \
    --timepoints teste,ovary,0hr,48hr,60hr,5day,10day \
    --pre-pde-timepoints teste,ovary,0hr,48hr,60hr \
    --post-pde-timepoints 5day,10day \
    --matrix-dir matrix_files_100kb \
    --eigenvector-dir .. \
    --global-scale \
    --plot-only \
    --output-dir output/
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--timepoints` | `teste,ovary,...,10day` | All timepoints in developmental order |
| `--pre-pde-timepoints` | `teste,ovary,0hr,48hr,60hr` | Pre-PDE timepoints |
| `--post-pde-timepoints` | `5day,10day` | Post-PDE timepoints |
| `--matrix-dir` | `matrix_files_100kb` | Base directory for Hi-C matrices |
| `--eigenvector-dir` | `.` | Parent directory for eigenvector subfolders |
| `--matrix-pattern` | `as_{}_iced_100kb.matrix` | Matrix filename pattern (`{}` = timepoint) |
| `--eigenvector-pattern` | `as_{}_iced_100kb.matrix.eigenvector` | Eigenvector filename pattern |
| `--max-bins` | 5000 | Maximum bins to analyze (0 = all) |
| `--pc1-threshold` | none | Exclude bins with \|PC1\| below threshold from switching analysis |
| `--global-scale` | off | Use same color scale across all saddle plots |
| `--plot-only` | off | Generate saddle plots only, skip switching/boundary analyses |
| `--boundary-linewidth` | 4.0 | A/B boundary line width on saddle plots |
| `--boundary-color` | yellow | A/B boundary line color on saddle plots |
| `--cbr-bed` | none | BED file of chromosome break regions |
| `--stage-names` | none | Display name mapping: `key1:val1,key2:val2,...` |

## Outputs

All outputs are written to `--output-dir`:

- `saddle_plot_{pde_stage}_{timepoint}.png/.svg` — individual saddle plots
- `saddle_strength_across_development.png/.svg` — strength trend line
- `saddle_strength_summary.csv` — per-timepoint strength metrics
- `saddle_strength_decomposition.png/.svg` — AA, BB, AB components
- `compartment_switching_*.png/.svg` — pairwise switching plots
- `sequential_compartment_switching.png/.svg` — stacked switching summary
- `boundary_strength.png/.svg` — boundary sharpness across development
- `compartment_entropy.png/.svg` — A/B transition frequency
- `compartment_strength.png/.svg` — mean |PC1| across development
- `*_cbr_comparison.*` — CBR-vs-genome plots (when `--cbr-bed` provided)

## Figure mapping

| Figure | Flags | Notes |
|--------|-------|-------|
| Fig. XE | `--global-scale --plot-only --boundary-color yellow` | Saddle plots, 7 stages |