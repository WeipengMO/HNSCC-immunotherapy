#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_hnscc_tcga_timp1_volcano")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TIMP1-high versus TIMP1-low differential expression.")
    parser.add_argument("--de-table", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--absolute-log2fc", type=float, default=1.0)
    parser.add_argument("--labels-per-direction", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.fdr <= 1 or args.absolute_log2fc < 0 or args.labels_per_direction < 0:
        raise ValueError("Invalid FDR, fold-change, or label limit")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    de = pd.read_csv(args.de_table, sep="\t")
    required = {"gene_id", "gene_name", "log2FoldChange", "pvalue", "padj"}
    missing = sorted(required - set(de.columns))
    if missing:
        raise ValueError(f"DE table is missing columns: {missing}")
    groups = pd.read_csv(args.groups, sep="\t", index_col=0)
    if not {"expression", "group"}.issubset(groups.columns):
        raise ValueError("Group table must contain expression and group")
    observed_groups = set(groups["group"].dropna().astype(str))
    if observed_groups != {"high", "low"}:
        raise ValueError(f"Expected high and low groups, found {sorted(observed_groups)}")

    de = de.replace([np.inf, -np.inf], np.nan).copy()
    de["plot_eligible"] = de[["gene_name", "log2FoldChange", "pvalue", "padj"]].notna().all(axis=1)
    de["minus_log10_pvalue"] = -np.log10(de["pvalue"].clip(lower=np.finfo(float).tiny))
    de["status"] = "Not significant"
    de.loc[de["plot_eligible"] & de["padj"].lt(args.fdr) &
           de["log2FoldChange"].ge(args.absolute_log2fc), "status"] = "Up"
    de.loc[de["plot_eligible"] & de["padj"].lt(args.fdr) &
           de["log2FoldChange"].le(-args.absolute_log2fc), "status"] = "Down"
    eligible = de.loc[de["plot_eligible"]].copy()
    label_frames = []
    for direction in ["Up", "Down"]:
        subset = eligible.loc[eligible["status"].eq(direction)].copy()
        subset["absolute_log2fc"] = subset["log2FoldChange"].abs()
        label_frames.append(subset.sort_values(["padj", "absolute_log2fc"], ascending=[True, False])
                            .drop_duplicates("gene_name").head(args.labels_per_direction))
    labels = pd.concat(label_frames, ignore_index=True) if label_frames else pd.DataFrame()
    de.to_csv(args.output_dir / "TIMP1_high_low_differential_expression.csv", index=False)
    eligible.to_csv(args.output_dir / "TIMP1_high_low_volcano_source_data.csv", index=False)
    labels.to_csv(args.output_dir / "TIMP1_high_low_volcano_labels.csv", index=False)
    groups.reset_index().to_csv(args.output_dir / "TIMP1_expression_groups.csv", index=False)
    group_summary = (groups.groupby("group", observed=True)
                     .agg(n_samples=("expression", "size"), expression_min=("expression", "min"),
                          expression_median=("expression", "median"), expression_max=("expression", "max"))
                     .reset_index())
    group_summary.to_csv(args.output_dir / "TIMP1_expression_group_summary.csv", index=False)
    pd.DataFrame([{"fdr": args.fdr, "absolute_log2fc": args.absolute_log2fc,
                   "labels_per_direction": args.labels_per_direction}]).to_csv(
        args.output_dir / "TIMP1_high_low_volcano_parameters.csv", index=False)

    colors = {"Not significant": "#BDBDBD", "Up": "#D95F02", "Down": "#2C7FB8"}
    fig, ax = plt.subplots(figsize=(3.5, 4))
    ax.grid(False)
    for status in ["Not significant", "Down", "Up"]:
        subset = eligible.loc[eligible["status"].eq(status)]
        ax.scatter(subset["log2FoldChange"], subset["minus_log10_pvalue"], s=8, alpha=0.45,
                   color=colors[status], label=status, linewidths=0, rasterized=True)
    ax.axvline(-args.absolute_log2fc, color="0.4", ls="--", lw=0.8)
    ax.axvline(args.absolute_log2fc, color="0.4", ls="--", lw=0.8)
    ax.axhline(-np.log10(args.fdr), color="0.4", ls="--", lw=0.8)
    ax.set_ylim(-0.02 * eligible["minus_log10_pvalue"].max(), 1.16 * eligible["minus_log10_pvalue"].max())
    label_texts = []
    for row in labels.itertuples(index=False):
        ax.scatter(row.log2FoldChange, row.minus_log10_pvalue, s=24, color=colors[row.status], zorder=3)
        label_texts.append(ax.text(row.log2FoldChange, row.minus_log10_pvalue, row.gene_name, fontsize=8))
    adjust_text(label_texts, ax=ax, expand_text=(1.15, 1.25), expand_points=(1.15, 1.25),
                force_text=(0.35, 0.55), force_points=(0.15, 0.25),
                arrowprops={"arrowstyle": "-", "color": "0.4", "lw": 0.45})
    ax.set(title="TIMP1 high vs low", xlabel="log2 fold change", ylabel="-log10(p-value)")
    ax.set_title("TIMP1 high vs low", pad=14)
    ax.legend(frameon=True, fontsize=8, loc="upper left")
    ax.tick_params(axis="both", which="major", length=4, width=0.8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "TIMP1_high_low_volcano.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.output_dir / "TIMP1_high_low_volcano.pdf", bbox_inches="tight")
    plt.close(fig)
    counts = eligible["status"].value_counts().to_dict()
    print(f"Plotted {len(eligible)} genes from {len(groups)} samples; status counts: {counts}")


if __name__ == "__main__":
    main()
