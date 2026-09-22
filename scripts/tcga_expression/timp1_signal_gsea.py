#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_hnscc_tcga_timp1_signal_gsea")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a declared TIMP1-associated pathway panel.")
    parser.add_argument("--gsea-table", type=Path, required=True)
    parser.add_argument("--pathways", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gsea = pd.read_csv(args.gsea_table, sep="\t")
    required_gsea = {"collection", "term", "nes", "pvalue", "padj", "leading_edge_genes"}
    missing_columns = sorted(required_gsea - set(gsea.columns))
    if missing_columns:
        raise ValueError(f"GSEA table is missing columns: {missing_columns}")
    pathways = pd.read_csv(args.pathways)
    required_pathways = {"term", "label"}
    if not required_pathways.issubset(pathways.columns):
        raise ValueError(f"Pathway table must contain: {sorted(required_pathways)}")
    if pathways["term"].duplicated().any():
        raise ValueError("Pathway terms must be unique")

    gsea.to_csv(args.output_dir / "TIMP1_high_low_gsea_all_pathways.csv", index=False)
    available = set(gsea["term"])
    missing = pathways.loc[~pathways["term"].isin(available)].copy()
    missing.to_csv(args.output_dir / "TIMP1_signal_pathways_missing.csv", index=False)
    if not missing.empty:
        raise ValueError(f"Configured pathways are missing from GSEA: {missing['term'].tolist()}")
    plotted = pathways.merge(gsea, on="term", how="left", validate="one_to_one")
    plotted["direction"] = np.where(plotted["nes"] < 0, "TIMP1-low enriched", "TIMP1-high enriched")
    plotted["minus_log10_fdr"] = -np.log10(plotted["padj"].clip(lower=np.finfo(float).tiny))
    plotted = plotted.sort_values("nes")
    plotted.to_csv(args.output_dir / "TIMP1_signal_gsea_source_data.csv", index=False)
    pathways.to_csv(args.output_dir / "TIMP1_signal_pathways_used.csv", index=False)

    limit = max(2.8, float(plotted["nes"].abs().max()))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, ax = plt.subplots(figsize=(3.2, 4.5))
    ax.scatter(plotted["nes"], plotted["label"],
               s=35 + 22 * plotted["minus_log10_fdr"].clip(upper=12),
               c=plotted["nes"], cmap="RdBu_r", norm=norm,
               edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="lightgray", linewidth=0.8)
    ax.set(xlabel="Normalized enrichment score", ylabel="", title="TIMP1 high vs low")
    size_breaks = [2, 4, 6, 8]
    handles = [ax.scatter([], [], s=35 + 22 * value, color="gray", edgecolor="white", linewidth=0.5)
               for value in size_breaks]
    ax.legend(handles, [str(value) for value in size_breaks], title="-log10(FDR)", frameon=False,
              loc="center left", bbox_to_anchor=(1.02, 0.5), labelspacing=1.2)
    fig.savefig(args.output_dir / "TIMP1_signal_gsea_nes.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.output_dir / "TIMP1_signal_gsea_nes.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Plotted all {len(plotted)} declared pathways from {len(gsea)} complete GSEA results")


if __name__ == "__main__":
    main()
