#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_hnscc_mif_pathway")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MIF-knockdown pathway reprogramming.")
    parser.add_argument("--gsea-table", type=Path, required=True)
    parser.add_argument("--terms", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--down-pathways", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.fdr <= 1 or args.down_pathways < 1:
        raise ValueError("--fdr must be in (0, 1] and --down-pathways must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gsea = pd.read_csv(args.gsea_table)
    terms = pd.read_csv(args.terms)
    required = {"collection", "term", "nes", "padj", "leading_edge_genes"}
    missing = sorted(required - set(gsea.columns))
    if missing:
        raise ValueError(f"GSEA table is missing columns: {missing}")
    if not {"group", "term", "label"}.issubset(terms.columns):
        raise ValueError("Term configuration must contain group, term and label")
    if terms["term"].duplicated().any():
        raise ValueError("Each configured term must occur only once")
    groups = {"negative_scrna_concordant", "positive_interferon_autophagy"}
    if not set(terms["group"]).issubset(groups):
        raise ValueError(f"Term group must be one of: {sorted(groups)}")

    gsea = gsea.copy()
    gsea["minus_log10_fdr"] = -np.log10(gsea["padj"].clip(lower=np.finfo(float).tiny))
    gsea["direction"] = np.where(gsea["nes"] > 0, "shMIF", "shCtrl")
    gsea.to_csv(args.output_dir / "pathway_reprogramming_all_gsea.csv", index=False)

    configured = terms.merge(gsea, on="term", how="left", validate="one_to_one", indicator=True)
    configured["present_in_gsea"] = configured["_merge"].eq("both")
    configured["passes_fdr"] = configured["padj"].le(args.fdr)
    configured["direction_matches"] = np.where(
        configured["group"].eq("negative_scrna_concordant"), configured["nes"].lt(0), configured["nes"].gt(0)
    )
    eligible = configured.loc[
        configured["present_in_gsea"] & configured["passes_fdr"] & configured["direction_matches"]
    ].copy()
    down_candidates = eligible.loc[eligible["group"].eq("negative_scrna_concordant")].sort_values("nes")
    down = down_candidates.head(args.down_pathways).copy()
    up = eligible.loc[eligible["group"].eq("positive_interferon_autophagy")].copy()
    selected_terms = set(down["term"]) | set(up["term"])
    configured["selected"] = configured["term"].isin(selected_terms)
    configured.drop(columns="_merge").to_csv(
        args.output_dir / "pathway_reprogramming_term_audit.csv", index=False
    )
    down_candidates.to_csv(args.output_dir / "pathway_reprogramming_down_candidates.csv", index=False)

    down["program"] = "scRNA-concordant programs reduced after shMIF"
    up["program"] = "Interferon and autophagy programs increased after shMIF"
    selected = pd.concat([down, up], ignore_index=True).sort_values("nes")
    selected.to_csv(args.output_dir / "pathway_reprogramming_plotted.csv", index=False)
    if selected.empty:
        raise ValueError("No configured pathways pass the FDR and direction criteria")

    sns.set_theme(style="whitegrid", context="notebook")
    limit = float(selected["nes"].abs().max())
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, ax = plt.subplots(figsize=(3, 6))
    ax.scatter(
        selected["nes"], selected["label"],
        s=45 + 22 * selected["minus_log10_fdr"].clip(upper=12),
        c=selected["nes"], cmap="RdBu_r", norm=norm,
        edgecolor="white", linewidth=0.65,
    )
    ax.axvline(0, color="lightgray", lw=1)
    ax.set_axisbelow(True)
    ax.grid(True, axis="both", color="#E0E0E0", linewidth=0.55, alpha=0.7)
    ax.invert_yaxis()
    ax.set(xlabel="Normalized enrichment score", ylabel="", title="THP1 response to SCC15 MIF knockdown")
    ax.xaxis.label.set_size(12)
    ax.title.set_size(12)
    size_breaks = [2, 4, 6]
    handles = [
        ax.scatter([], [], s=45 + 22 * value, color="#888888", edgecolor="white", linewidth=0.6)
        for value in size_breaks
    ]
    ax.legend(
        handles, [str(value) for value in size_breaks], title="-log10(FDR)", frameon=False,
        loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0, labelspacing=1.0,
    )
    fig.savefig(args.output_dir / "pathway_reprogramming.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output_dir / "pathway_reprogramming.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Plotted {len(down)} down-shifted and {len(up)} up-shifted configured pathways")


if __name__ == "__main__":
    main()
