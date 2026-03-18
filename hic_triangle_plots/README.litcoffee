# triangle_heatmaps

Generate stacked triangular Hi-C heatmaps across developmental timepoints for
*Parascaris univalens*.  Each timepoint is rendered as an upward-pointing
triangle and stacked vertically, allowing visual comparison of chromatin
interaction patterns through embryonic development and programmed DNA
elimination (PDE).

## Overview

`plot_hic_stages.py` extracts a specified genomic region from ICE-normalized
Hi-C matrices at each timepoint and renders them as stacked triangle heatmaps.
Supports both raw contact frequency and observed/expected (O/E) visualization.

For post-PDE timepoints, the script handles the coordinate system switch from
germline to somatic genome: the `--region` argument specifies somatic chromosome
names (e.g., `chrX7-chrX9`), which are mapped back to germline coordinates for
pre-PDE matrix extraction and used directly for post-PDE somatic matrix
extraction.  O/E expected values are calculated chromosome-aware for post-PDE
samples using somatic chromosome boundaries.

Outputs per plot type (PNG, TIFF, SVG):
  - PNG at high resolution (DPI scaled to bin count)
  - TIFF with LZW compression
  - SVG with editable text for Adobe Illustrator

## Dependencies

```
numpy
matplotlib
scipy
```

## Input files

Three reference files are needed (shared with `interaction_scoring/`):

| File | Description |
|------|-------------|
| `data/pu_v2_germ_to_soma_mapping.bed` | Germline-to-somatic chromosome mapping: `germ_chr  start  end  soma_chr` |
| `data/{resolution}/pu_v2_prepde_abs_{res}` | HiC-Pro abs.bed for germline genome at the target resolution |
| `data/{resolution}/pu_v2_postpde_abs_{res}` | HiC-Pro abs.bed for somatic genome at the target resolution |

Hi-C matrices used by this script are ICE-normalized sparse matrices generated
by [HiC-Pro](https://github.com/nservant/HiC-Pro).  Raw sequencing reads are
available from SRA under accession SRPXXXXXX.  To regenerate matrices, run
HiC-Pro with the Parascaris v2 germline reference genome, then apply ICE
balancing.  The script expects them at `matrix_files_{res}/prepde/` and
`matrix_files_{res}/postpde/` with the naming pattern
`pu_{timepoint}_{resolution}_iced.matrix` (see `get_matrix_files`-style logic
in `main()` to modify).

## Usage

### log2(O/E) + raw — both plot types, 40kb resolution

```bash
python plot_hic_stages.py \
    --region chrX7-chrX9 \
    --mapping data/pu_v2_germ_to_soma_mapping.bed \
    --resolution 40000 \
    --bin-bed data/40000/pu_v2_prepde_abs_40kb \
    --soma-bin-bed data/40000/pu_v2_postpde_abs_40kb \
    --timepoints '10hr,17hr,24hr,48hr,72hr' \
    --plot-type both \
    --vmax 15 \
    --vmin 0.0 \
    --vmax-obsexp 2.5 \
    --log2-obsexp \
    --no-outline \
    --output-dir results/
```

### Raw only, direct coordinate range, 20kb resolution

```bash
python plot_hic_stages.py \
    --region 10000000-15000000 \
    --mapping data/pu_v2_germ_to_soma_mapping.bed \
    --resolution 20000 \
    --bin-bed data/20000/pu_v2_prepde_abs_20kb \
    --timepoints '10hr,17hr,24hr,48hr,72hr' \
    --plot-type raw \
    --vmax 100 \
    --output-dir results/
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--region` | `10000000-15000000` | Region to plot: somatic chromosome range (`chrX7-chrX9`) or direct germline coordinates |
| `--resolution` | 20000 | Hi-C bin resolution in bp |
| `--plot-type` | both | `raw`, `obsexp`, or `both` |
| `--vmax` | 100 | Color scale max for raw plots |
| `--vmin` | 0.1 | Color scale min for raw plots |
| `--vmax-obsexp` | 3.0 | Color scale max for O/E plots (for log2: symmetric, e.g. 2.5 = range -2.5 to +2.5) |
| `--vmin-obsexp` | 0.5 | Color scale min for O/E plots (ignored if `--log2-obsexp`) |
| `--log2-obsexp` | off | Apply log2 to O/E values (symmetric scale around 0) |
| `--no-normalize` | — | Disable median-contact normalization for raw plots |
| `--no-outline` | — | Disable black triangle outlines |
| `--timepoints` | `10hr,17hr,24hr,48hr,72hr` | Comma-separated timepoints, or `all` |

## Outputs

Files are written to `--output-dir` with the naming pattern
`hic_developmental{type_suffix}_{start}_{end}.{ext}`:

- `.png` — high-resolution raster
- `.tiff` — LZW-compressed TIFF
- `.svg` — editable text for Illustrator

## Figure mapping

| Figure | Region | Resolution | Flags | Notes |
|--------|--------|-----------|-------|-------|
| Fig. XB | `chrX7-chrX9` | 40kb | `--plot-type both --log2-obsexp --vmax 15 --vmax-obsexp 2.5` | |