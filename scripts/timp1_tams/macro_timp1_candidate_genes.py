#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_hnscc_macro_timp1_candidates")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the Macro-TIMP1 candidate-gene panel by response.")
    parser.add_argument("--expression", type=Path, required=True, help="Patient-by-gene log2 CPM table.")
    parser.add_argument("--metadata", type=Path, required=True, help="Patient metadata table indexed like the expression table.")
    parser.add_argument("--de-table", type=Path, required=True, help="Full NR-versus-R differential-expression table.")
    parser.add_argument("--gene-panel", type=Path, required=True, help="CSV with panel and gene columns in plotting order.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def map_sizes(values: pd.Series, minimum: float, maximum: float) -> np.ndarray:
    if maximum == minimum:
        return np.full(len(values), (25 + 420) / 2)
    return np.interp(values, [minimum, maximum], [25, 420])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expression = pd.read_csv(args.expression, index_col=0)
    metadata = pd.read_csv(args.metadata, index_col=0)
    differential = pd.read_csv(args.de_table, index_col=0)
    panel = pd.read_csv(args.gene_panel)

    required_panel = {"panel", "gene"}
    missing_panel = sorted(required_panel - set(panel.columns))
    if missing_panel:
        raise ValueError(f"Gene-panel table is missing columns: {', '.join(missing_panel)}")
    if "response" not in metadata:
        raise ValueError("Metadata table does not contain a response column")
    if "pval" not in differential:
        raise ValueError("Differential-expression table does not contain a pval column")
    if panel["gene"].duplicated().any():
        duplicates = panel.loc[panel["gene"].duplicated(), "gene"].tolist()
        raise ValueError(f"Genes occur more than once in the panel: {duplicates}")

    shared_patients = expression.index.intersection(metadata.index)
    metadata = metadata.loc[shared_patients]
    expression = expression.loc[shared_patients]
    keep_patients = metadata["response"].isin(["R", "NR"])
    metadata = metadata.loc[keep_patients]
    expression = expression.loc[metadata.index]
    available = panel[
        panel["gene"].isin(expression.columns) & panel["gene"].isin(differential.index)
    ].copy()
    omitted = panel.loc[~panel["gene"].isin(available["gene"]), ["panel", "gene"]].copy()
    omitted.to_csv(args.output_dir / "candidate_gene_panel_omitted_genes.csv", index=False)
    if available.empty:
        raise ValueError("None of the candidate-panel genes are available in both input tables")

    genes = available["gene"].tolist()
    group_mean = expression[genes].groupby(metadata["response"]).mean().T.reindex(columns=["R", "NR"])
    sample_z = expression[genes].apply(
        lambda values: (values - values.mean()) / values.std(ddof=1)
        if values.std(ddof=1) != 0 else pd.Series(0.0, index=values.index),
        axis=0,
    )
    group_z = sample_z.groupby(metadata["response"]).mean().T.reindex(columns=["R", "NR"])
    minus_log10_p = -np.log10(differential.loc[genes, "pval"].clip(lower=1e-300))
    size_min = float(minus_log10_p.min())
    size_max = float(minus_log10_p.max())

    records = []
    for row in available.itertuples(index=False):
        for response in ["R", "NR"]:
            records.append({
                "panel": row.panel,
                "gene": row.gene,
                "response": response,
                "mean_log2cpm": group_mean.loc[row.gene, response],
                "mean_sample_zscore": group_z.loc[row.gene, response],
                "minus_log10_nominal_p": minus_log10_p.loc[row.gene],
            })
    plot_data = pd.DataFrame(records)
    plot_data["dot_size"] = map_sizes(plot_data["minus_log10_nominal_p"], size_min, size_max)
    plot_data.to_csv(args.output_dir / "integrated_R_NR_candidate_gene_dot_heatmap_data.csv", index=False)
    available.to_csv(args.output_dir / "integrated_R_NR_candidate_gene_panel_used.csv", index=False)

    panels = available[["panel"]].drop_duplicates()
    z_limit = max(1, math.ceil(float(np.nanmax(np.abs(plot_data["mean_sample_zscore"])))))
    norm = mpl.colors.TwoSlopeNorm(vmin=-z_limit, vcenter=0, vmax=z_limit)
    cmap = sns.color_palette("vlag", as_cmap=True)
    width_ratios = [max(0.9, 0.11 * len(available.loc[available["panel"].eq(name)])) for name in panels["panel"]] + [0.7]
    fig = plt.figure(figsize=(7, 4))
    grid = fig.add_gridspec(1, len(panels) + 1, width_ratios=width_ratios, wspace=0.38)
    axes = [fig.add_subplot(grid[0, index]) for index in range(len(panels))]
    legend_ax = fig.add_subplot(grid[0, len(panels)])
    legend_ax.axis("off")

    scatter = None
    x_map = {"R": 0, "NR": 0.72}
    for axis, panel_name in zip(axes, panels["panel"]):
        genes_in_panel = available.loc[available["panel"].eq(panel_name), "gene"].tolist()
        plot_order = genes_in_panel[::-1]
        y_map = {gene: index for index, gene in enumerate(plot_order)}
        subset = plot_data.loc[plot_data["panel"].eq(panel_name)].copy()
        subset["x"] = subset["response"].map(x_map)
        subset["y"] = subset["gene"].map(y_map)
        scatter = axis.scatter(subset["x"], subset["y"], s=subset["dot_size"],
                               c=subset["mean_sample_zscore"], cmap=cmap, norm=norm,
                               linewidths=0.45, edgecolors="white")
        axis.set_xticks([0, 0.72])
        axis.set_xticklabels(["R", "NR"], fontsize=9.5)
        axis.set_yticks(range(len(plot_order)))
        axis.set_yticklabels(plot_order, fontsize=9.5)
        axis.set_xlim(-0.4, 1.2)
        axis.set_ylim(-0.6, len(plot_order) - 0.4)
        axis.set_title(panel_name, fontsize=11.5, pad=7)
        for x_value in [0, 0.72]:
            axis.axvline(x_value, color="#F0F0F0", linewidth=0.9, zorder=0)
        sns.despine(ax=axis, top=True, right=True)
        axis.grid(False)

    colorbar_ax = legend_ax.inset_axes([-0.25, 0.60, 0.20, 0.28])
    colorbar = fig.colorbar(scatter, cax=colorbar_ax)
    colorbar.set_label("z-score", fontsize=8.5, labelpad=5)
    colorbar.ax.tick_params(labelsize=7.8)
    colorbar.set_ticks(np.linspace(-z_limit, z_limit, 5))
    legend_values = np.linspace(size_min, size_max, 5)
    legend_sizes = map_sizes(pd.Series(legend_values), size_min, size_max)
    handles = [axes[0].scatter([], [], s=size, facecolor="#4D4D4D", edgecolor="white", linewidth=0.4) for size in legend_sizes]
    legend_ax.legend(handles, [f"{value:.1f}" for value in legend_values], title=r"-$\log_{10}$(p)",
                     loc="upper left", bbox_to_anchor=(-0.5, 0.43), frameon=False,
                     title_fontsize=8.5, fontsize=7.8, labelspacing=0.8, handletextpad=0.7)
    fig.subplots_adjust(left=0.09, right=1, top=0.90, bottom=0.12)
    fig.savefig(args.output_dir / "integrated_R_NR_candidate_gene_dot_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.output_dir / "integrated_R_NR_candidate_gene_dot_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Plotted {len(available)} genes from {len(panels)} groups; {len(omitted)} genes omitted")


if __name__ == "__main__":
    main()
