# HNSCC immunotherapy analysis code

This repository contains the analysis code used to generate results
for the study of head and neck squamous cell carcinoma (HNSCC) immunotherapy.

## Repository structure

```text
.
├── config/                     # Ordered biological panels and gene-set definitions
├── scripts/
│   ├── baseline_ecosystem/     # Pretreatment patient-level ecosystem composition
│   ├── malignant_state_transition/ # Malignant meta-program discovery, overlap and combinations
│   ├── mif_knockdown/          # THP1 response to SCC15 MIF knockdown
│   ├── single_cell_landscape/  # Patient-level single-cell signature summaries
│   ├── spatial_niche/          # Spatial TIMP1+ TAM neighborhood analyses
│   ├── tcga_expression/        # TCGA-HNSCC TIMP1 expression analyses
│   └── timp1_tams/             # TIMP1+ TAM pathways, genes and regulons
├── results/                    # Reproduced figures and source-data tables
├── environment.yml            # Conda environment used for reproduction
└── requirements.txt           # Equivalent direct Python dependencies
```

## Environment setup

The scripts were validated with Python 3.10 on Linux. The recommended setup is
Conda or Mamba:

```bash
conda env create -f environment.yml
conda activate hnscc-immunotherapy
```

Alternatively, create a virtual environment and install the direct Python
dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export MPLCONFIGDIR=/tmp/mplconfig_hnscc_repository
export NUMBA_CACHE_DIR=/tmp/numba_cache_hnscc_repository
```