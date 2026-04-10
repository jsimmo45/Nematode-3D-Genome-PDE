# trans_cbr_profile

Analyze trans (inter-chromosomal) Hi-C interaction profiles around chromosome
break regions (CBRs) in *Ascaris*.  Quantifies how trans contact frequency
changes approaching PDE break sites, and decomposes the signal to show which
partner chromosomes contribute to the enrichment.

## Overview

`trans_cbr_profile.py` reads trans-only allValidPairs files from HiC-Pro,
bins contacts at 5 kb resolution flanking each CBR, and produces:

- **Combined metaprofiles** (mean ± SEM across CBRs) for end-type and
  internal break regions separately
- **Component breakdown** — stacked area plot showing per-partner-chromosome
  contributions to the trans signal (the "which chromosomes are interacting"
  question)
- **Per-CBR individual profiles** with data tables
- **Grid overview** showing all CBRs in a single figure
- **Selected sample comparisons** with optional derivative curves

## Dependencies

```
numpy
pandas
matplotlib
```

## Input files

| File | In repo? | Description |
|------|----------|-------------|
| `data/AG_v50_chrom.bed` | Yes | Chromosome sizes BED for the pre-PDE genome |
| `data/cbr_v50_500kb_windows_cbr.bed` | Yes | CBR positions (chr start end orientation name type) |
| Trans allValidPairs | No | Trans-only HiC-Pro pairs files; regenerate from SRA (SRPXXXXXX) |

## Usage

```bash
python trans_cbr_profile.py \
    --pre_bed data/AG_v50_chrom.bed \
    --pre_samples \
        allValidPairs/AG.v50.teste.trans.allValidPairs \
        allValidPairs/AG.v50.48hr.trans.allValidPairs \
        allValidPairs/AG.v50.60hr.trans.allValidPairs \
    --post_samples \
        allValidPairs/AG.v50.5day.trans.allValidPairs \
    --bed data/cbr_v50_500kb_windows_cbr.bed \
    --binsize 5000 \
    --flank 500000 \
    --normalize \
    --components teste \
    --component_type end \
    --dpi 300 \
    --output_dir results/ \
    --output_table trans_cbr_profile.txt \
    --output_plot trans_cbr_profile.png \
    --output_plot_end trans_cbr_profile_end.png \
    --output_plot_internal trans_cbr_profile_internal.png \
    --grid_plot cbrs_trans_grid.png
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--pre_bed` | required | Pre-PDE chromosome BED file |
| `--pre_samples` | required | Pre-PDE trans allValidPairs files |
| `--post_samples` | none | Post-PDE trans allValidPairs files |
| `--bed` | required | CBR BED file (chr start end orientation name type) |
| `--binsize` | 5000 | Bin size in bp |
| `--flank` | 250000 | Flank distance from CBR center |
| `--normalize` | off | Apply CPM normalization |
| `--components` | none | Sample name substring for component analysis |
| `--component_type` | all | CBR type for components: `end`, `internal`, or `all` |
| `--selected_samples` | none | Sample substrings for comparison plots |
| `--derivative_samples` | none | Samples to add derivative curves |
| `--dpi` | 600 | Output resolution |
| `--ylim` | auto | Y-axis maximum |

## Outputs

In `--output_dir`:
- `trans_cbr_profile.png/.tiff/.svg` — combined metaprofile (all CBRs)
- `trans_cbr_profile_end.png` — end-type CBRs only
- `trans_cbr_profile_internal.png` — internal CBRs only
- `cbrs_trans_grid.png` — grid of all individual CBRs
- `individual_cbrs/` — per-CBR plots and data tables
- `components/combined_components_{type}.png` — chromosome component breakdown
- `components/*_trans_components.png` — per-CBR component plots

## Figure mapping

| Figure | Flags | Notes |
|--------|-------|-------|
| Fig. XK | `--component_type end --components teste --flank 500000` | End CBRs, 5kb bins |
