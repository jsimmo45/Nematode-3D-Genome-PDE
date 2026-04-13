# interaction_scoring

Quantify and visualize Hi-C interaction frequencies between pairs of genomic
regions across developmental timepoints in *Parascaris univalens*.  Designed for
organisms that undergo **programmed DNA elimination (PDE)**, where a single
germline chromosome fragments into multiple somatic chromosomes after
fertilization.

## Overview

`plot_interaction_heatmap.py` takes a BED file of interacting region pairs and
ICE-normalized Hi-C matrices (HiC-Pro format), computes interaction scores for
each region pair at each timepoint, and produces:

- **Region × timepoint heatmap** showing interaction dynamics across development
- **Summary dot plot** with mean/median ± SEM, optional Mann-Whitney U
  significance brackets vs. a reference timepoint
- **CSV matrix** of all interaction scores for downstream analysis

### Normalization

The recommended mode is **log2(O/E)** (`--oe-normalize --log2-oe`), which
computes chromosome-aware expected contact frequencies at each genomic distance,
divides observed by expected, and applies a log2 transform.  This centers the
scale around 0 (enriched = positive, depleted = negative) and controls for the
distance-decay inherent to Hi-C data.

Key methodological details:

- Expected values are calculated **per chromosome**, not across the whole matrix.
  This prevents inter-chromosomal zeros from contaminating the distance-dependent
  expected profile.
- For **post-PDE timepoints**, germline bins are grouped by their somatic
  chromosome assignment to correctly distinguish intra- vs. inter-chromosomal
  contacts.
- All matrix lookups use **germline bin coordinates** (the matrices are in
  germline space), while the germline-to-somatic mapping is used only to
  determine whether a region pair is intra- or inter-chromosomal after PDE.
- CPM normalization is redundant with O/E (the scalar cancels in the ratio) and
  is automatically disabled when both flags are set.

## Dependencies

```
numpy
pandas
matplotlib
seaborn
scipy
```

## Input files

Four reference files are included in this repository under `data/`:

| File | Description |
|------|-------------|
| `data/interaction_regions/interacting_regions_100kb.bed` | Region pairs to score: `chr  start1  end1  start2  end2  [name]` |
| `data/20000/pu_v2_prepde_abs_20kb` | HiC-Pro abs.bed mapping bin numbers → germline coordinates (20 kb resolution) |
| `data/20000/pu_v2_postpde_abs_20kb` | HiC-Pro abs.bed for the somatic genome assembly (20 kb resolution) |
| `data/pu_v2_germ_to_soma_mapping.bed` | Germline-to-somatic chromosome mapping: `germ_chr  start  end  soma_chr` |

Hi-C matrices used by this script are ICE-normalized sparse matrices generated
by [HiC-Pro](https://github.com/nservant/HiC-Pro) at 20 kb resolution.  Raw
sequencing reads are available from [GSE315650](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE315650).  To
regenerate matrices, run HiC-Pro with the Parascaris v2 germline reference
genome, then apply ICE balancing.  The script expects matrices at
`matrix_files_{res}/prepde/` and `matrix_files_{res}/postpde/` with the naming
pattern `pu_{timepoint}_{resolution}_iced.matrix` (see `get_matrix_files()` to
modify).

## Usage

### log2(O/E) — recommended for publication figures

```bash
python plot_interaction_heatmap.py \
    --regions data/interaction_regions/interacting_regions_100kb.bed \
    --timepoints '10hr,17hr,24hr,48hr,72hr' \
    --resolution 20000 \
    --bin-bed data/20000/pu_v2_prepde_abs_20kb \
    --soma-bin-bed data/20000/pu_v2_postpde_abs_20kb \
    --germ-to-soma-mapping data/pu_v2_germ_to_soma_mapping.bed \
    --oe-normalize \
    --log2-oe \
    --method mean \
    --color-dots \
    --ref-timepoint 17hr \
    --output interaction_heatmap_100kb \
    --output-dir results/
```

### Raw interaction frequencies (no normalization)

```bash
python plot_interaction_heatmap.py \
    --regions data/interaction_regions/interacting_regions_100kb.bed \
    --timepoints '10hr,17hr,24hr,48hr,72hr' \
    --resolution 20000 \
    --bin-bed data/20000/pu_v2_prepde_abs_20kb \
    --no-normalize \
    --output interaction_heatmap_raw \
    --output-dir results/
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--resolution` | 20000 | Hi-C bin resolution in bp |
| `--method` | mean | Scoring method: `mean`, `max`, `sum`, `percentile95` |
| `--oe-normalize` | off | Compute observed/expected normalization |
| `--log2-oe` | off | Apply log2 to O/E values (requires `--oe-normalize`) |
| `--cpm` | off | Counts-per-million normalization (redundant with O/E) |
| `--no-normalize` | — | Disable default total-contact normalization |
| `--summary-stat` | both | Overlay on summary plot: `mean`, `median`, or `both` |
| `--color-dots` | off | Color individual dots by RdBu_r scale on summary plot |
| `--ref-timepoint` | none | Reference timepoint for Mann-Whitney U tests (e.g. `17hr`) |
| `--vmin` / `--vmax` | auto | Heatmap color scale limits |
| `--ymax` | auto | Summary plot y-axis maximum |

## Outputs

All outputs are written to `--output-dir` with `--output` as the filename prefix:

- `{prefix}.png` / `{prefix}.svg` — interaction heatmap
- `{prefix}_summary_{stat}.png` / `.svg` — summary dot+errorbar plot
- `{prefix}.csv` — interaction score matrix

SVG files use editable text (`svg.fonttype = 'none'`) for downstream editing in
Adobe Illustrator.

## Figure mapping

| Figure | Regions file | Flags | Notes |
|--------|-------------|-------|-------|
| Fig. 6D, 6E | `interacting_regions_100kb.bed` | `--oe-normalize --log2-oe --ref-timepoint 17hr` | |
