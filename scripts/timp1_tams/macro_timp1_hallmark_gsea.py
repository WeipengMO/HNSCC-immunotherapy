#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_hnscc_macro_timp1_hallmark")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Hallmark pathways enriched in non-responders.")
    parser.add_argument("--gsea-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be a positive integer")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gsea = pd.read_csv(args.gsea_table)
    required = {"Term", "NES", "FDR q-val"}
    missing = sorted(required - set(gsea.columns))
    if missing:
        raise ValueError(f"GSEA table is missing columns: {', '.join(missing)}")

    gsea["NES"] = pd.to_numeric(gsea["NES"], errors="coerce")
    gsea["FDR q-val"] = pd.to_numeric(gsea["FDR q-val"], errors="coerce")
    all_nr = gsea.loc[gsea["NES"].gt(0)].copy()
    all_nr["minus_log10_fdr"] = -np.log10(all_nr["FDR q-val"].clip(lower=1e-300))
    all_nr = all_nr.sort_values(["FDR q-val", "NES"], ascending=[True, False])
    selected = all_nr.head(args.top_n).sort_values("NES", ascending=False).copy()
    all_nr.to_csv(args.output_dir / "hallmark_gsea_NR_direction_all_pathways.csv", index=False)
    selected.to_csv(args.output_dir / "hallmark_gsea_NR_direction_plotted_pathways.csv", index=False)

    fdr_min = float(selected["minus_log10_fdr"].min())
    fdr_max = float(selected["minus_log10_fdr"].max())
    if fdr_min == fdr_max:
        dot_sizes = np.full(len(selected), 125.0)
    else:
        dot_sizes = np.interp(selected["minus_log10_fdr"], [fdr_min, fdr_max], [30, 220])
    nes_norm = mpl.colors.Normalize(vmin=float(selected["NES"].min()), vmax=float(selected["NES"].max()))
    base_cmap = mpl.colormaps["Reds"]
    nes_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "Reds_truncated", base_cmap(np.linspace(0.35, 1.0, 256))
    )
    y_positions = np.arange(len(selected))[::-1]
    fig, ax = plt.subplots(figsize=(2.5, 4))
    ax.scatter(selected["NES"], y_positions, s=dot_sizes, c=selected["NES"],
               cmap=nes_cmap, norm=nes_norm, linewidth=0, zorder=3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(selected["Term"])
    ax.axvline(0, color="#777777", linewidth=0.7, linestyle="--")
    ax.set_title("NR enriched")
    ax.set_xlabel("Normalized enrichment score")
    ax.set_ylabel("")
    legend_values = np.linspace(fdr_min, fdr_max, 4)
    legend_sizes = np.full(4, 125.0) if fdr_min == fdr_max else np.interp(legend_values, [fdr_min, fdr_max], [30, 220])
    handles = [ax.scatter([], [], s=size, color="#7d7d7d", linewidth=0) for size in legend_sizes]
    ax.legend(handles, [f"{value:.1f}" for value in legend_values], title=r"-log10(FDR)",
              bbox_to_anchor=(1.02, 0.75), loc="upper left", frameon=False)
    ax.set_axisbelow(True)
    ax.grid(True, axis="both", color="#E6E6E6", linestyle="--", linewidth=0.7)
    fig.savefig(args.output_dir / "hallmark_gsea_NR_direction_dotplot.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.output_dir / "hallmark_gsea_NR_direction_dotplot.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Selected the top {len(selected)} of {len(all_nr)} NR-enriched Hallmark pathways by FDR")


if __name__ == "__main__":
    main()
