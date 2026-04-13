# clustering

Hierarchical clustering of whole-chromosome Hi-C interaction frequencies in
*Ascaris suum* across developmental timepoints.  Generates clustered heatmaps
showing inter-chromosomal contact patterns before and after programmed DNA
elimination (PDE).

## Overview

`clustering_heatmap.py` reads HiC-Pro allValidPairs files, counts
inter-chromosomal contacts, builds a chromosome-by-chromosome interaction
matrix, and performs hierarchical clustering with heatmap visualization.

Pre-PDE and post-PDE samples are handled separately because the chromosome
count changes after DNA elimination (the germline genome has fewer, larger
chromosomes; the somatic genome has more, smaller chromosomes).  Each gets its
own chromosome order file and configurable label font size.

### Normalization

Three modes:

- **McCord expected-trans** (`--mccord`, recommended): normalizes each
  chromosome pair by its expected contact frequency based on row marginals.
  Corrects for differences in chromosome size and total contacts.
- **Row-max** (`--normalize`): divides each row by its maximum value.
- **Raw** (default): no normalization, uses symmetric raw counts.

## Dependencies

```
numpy
pandas
seaborn
matplotlib
scipy
```

## Input files

| File | In repo? | Description |
|------|----------|-------------|
| `data/chrom_order_pre.txt` | Yes | Chromosome display order for pre-PDE (one name per line) |
| `data/chrom_order_post.txt` | Yes | Chromosome display order for post-PDE |
| allValidPairs files | No | HiC-Pro output; regenerate from GEO [GSE314626] (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE314626). |

allValidPairs files are large intermediate outputs from HiC-Pro (column 2 =
query chromosome, column 5 = target chromosome).  Raw sequencing reads are
available from SRA under accession SRPXXXXXX.

## Usage

```bash
python clustering_heatmap.py \
    --pre_pde_samples \
        allValidPairs/AG.v50.1cell.allValidPairs \
        allValidPairs/AG.v50.48hr.allValidPairs \
        allValidPairs/AG.v50.60hr.allValidPairs \
    --post_pde_samples \
        allValidPairs/AG.v50.5day.allValidPairs \
        allValidPairs/AG.v50.10day.allValidPairs \
    --order_pre data/chrom_order_pre.txt \
    --order_post data/chrom_order_post.txt \
    --mccord \
    --thresh 2.0 \
    --label_fontsize_pre 22 \
    --label_fontsize_post 16 \
    --hide_row_dendrogram \
    --dpi 300 \
    --output_dir results/ \
    --output_clusters clusters.bed \
    --output_heatmap heatmap.png \
    --output_table sorted.xlsx \
    --output_matrix matrix.xlsx \
    --output_combined combined.xlsx \
    --output_pca pca.png \
    --output_cluster_heatmap cluster_heatmap.png
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--pre_pde_samples` | required | Pre-PDE allValidPairs files (space-separated) |
| `--post_pde_samples` | none | Post-PDE allValidPairs files (space-separated) |
| `--order_pre` | required | Chromosome order file for pre-PDE |
| `--order_post` | required | Chromosome order file for post-PDE |
| `--mccord` | off | Use expected-trans (McCord) normalization |
| `--normalize` | off | Use row-max normalization |
| `--thresh` | auto | Clustering distance threshold for `fcluster` |
| `--nclusters` | none | Alternative: specify exact number of clusters |
| `--label_fontsize_pre` | 16 | Font size for pre-PDE heatmap labels |
| `--label_fontsize_post` | 12 | Font size for post-PDE heatmap labels |
| `--hide_row_dendrogram` | off | Hide left-side row dendrogram |
| `--dpi` | 600 | Output resolution |
| `--target` | none | Restrict to specific chromosomes (comma-separated) |

## Outputs

Per sample (in `--output_dir`):
- `{sample}_heatmap.png` / `.pdf` — clustered interaction heatmap
- `{sample}_clusters.bed` — cluster assignments (TSV)

Color scale is shared across all samples for direct visual comparison.

## Figure mapping

| Figure | Samples | Flags | Notes |
|--------|---------|-------|-------|
| Fig. 4C | 1cell, 48hr, 60hr (pre); 5day, 10day (post) | `--mccord --thresh 2.0 --hide_row_dendrogram` | Global color scale |
