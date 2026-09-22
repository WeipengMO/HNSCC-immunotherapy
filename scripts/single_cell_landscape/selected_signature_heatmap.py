#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from scipy import sparse
import seaborn as sns


GROUP_ORDER = ["Pre-R", "Pre-NR", "Post-R", "Post-NR"]

CELL_TYPE_COLORS = {
    "T": "#e1812c",
    "Macro / DC": "#aa40fc",
    "Fibroblast": "#17becf",
    "Epithelial (malig.)": "#98df8a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the selected cell-state signature heatmap.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing *.ICB_Chemo.h5ad files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for figure and source-data outputs.")
    parser.add_argument("--signature-definitions", type=Path, required=True)
    parser.add_argument("--figure-panel", type=Path, required=True)
    parser.add_argument("--layer", default="log1p_norm", help="AnnData expression layer (default: log1p_norm).")
    parser.add_argument("--min-cells", type=int, default=20, help="Minimum cells per patient/time/cell-type group (default: 20).")
    parser.add_argument("--min-genes", type=int, default=3, help="Minimum observed genes per signature (default: 3).")
    return parser.parse_args()


def calculate_patient_gene_means(files: list[Path], layer: str, min_cells: int, selected_genes: list[str]) -> pd.DataFrame:
    required_obs = ["patient", "treatment", "response", "cell_type_l1"]
    rows: list[dict[str, object]] = []

    for path in files:
        adata = ad.read_h5ad(path, backed="r")
        try:
            missing_obs = sorted(set(required_obs) - set(adata.obs.columns))
            if missing_obs:
                raise ValueError(f"{path.name} is missing obs columns: {', '.join(missing_obs)}")
            if layer not in adata.layers:
                raise ValueError(f"{path.name} does not contain the {layer!r} layer")

            genes = [gene for gene in selected_genes if gene in adata.var_names]
            obs = adata.obs[required_obs].astype("string")
            keep = obs["response"].isin(["R", "NR"]) & obs["treatment"].isin(["pre", "post"])
            obs = obs.loc[keep]
            expression = adata[keep.to_numpy(), genes].layers[layer]
            expression = expression.tocsr() if sparse.issparse(expression) else np.asarray(expression)

            grouped = obs.groupby(required_obs, observed=True, sort=False).indices
            for keys, indices in grouped.items():
                if len(indices) < min_cells:
                    continue
                row = dict(zip(required_obs, map(str, keys)))
                row["n_cells"] = len(indices)
                row.update(zip(genes, np.asarray(expression[indices].mean(axis=0)).ravel()))
                rows.append(row)
        finally:
            adata.file.close()

    if not rows:
        raise ValueError("No cell groups passed the response, treatment, and minimum-cell filters")
    return pd.DataFrame(rows)


def calculate_signature_scores(gene_means: pd.DataFrame, definitions: pd.DataFrame, min_genes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = ["patient", "treatment", "response", "cell_type_l1", "n_cells"]
    score_tables: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []

    for row in definitions.itertuples(index=False):
        cell_type, signature = row.cell_type_l1, row.signature
        requested_genes = str(row.genes).split(";")
        subset = gene_means.loc[gene_means["cell_type_l1"].eq(cell_type)]
        genes = [gene for gene in requested_genes if gene in subset and subset[gene].notna().any()]
        coverage_rows.append({
            "cell_type_l1": cell_type,
            "signature": signature,
            "n_requested": len(requested_genes),
            "n_observed": len(genes),
            "genes": ";".join(genes),
        })
        if len(genes) < min_genes:
            continue
        scores = subset[metadata].copy()
        scores["signature"] = signature
        scores["score"] = subset[genes].mean(axis=1)
        scores["group"] = scores["treatment"].str.capitalize() + "-" + scores["response"]
        scores["n_genes"] = len(genes)
        score_tables.append(scores)

    if not score_tables:
        raise ValueError("No selected signature met the minimum observed-gene threshold")
    return pd.concat(score_tables, ignore_index=True), pd.DataFrame(coverage_rows)


def make_heatmap_matrix(scores: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_order = (panel["cell_type_l1"] + " | " + panel["signature"]).tolist()
    scores = scores.copy()
    scores["feature"] = scores["cell_type_l1"] + " | " + scores["signature"]
    group_means = (
        scores.groupby(["feature", "group"], observed=True)["score"]
        .mean().unstack().reindex(index=feature_order, columns=GROUP_ORDER)
    )
    if group_means.isna().any().any():
        missing = group_means.isna().stack().loc[lambda values: values].index.tolist()
        raise ValueError(f"Missing group means for selected signatures: {missing}")
    row_sd = group_means.std(axis=1, ddof=0).replace(0, np.nan)
    z_scores = group_means.sub(group_means.mean(axis=1), axis=0).div(row_sd, axis=0).T
    return group_means, z_scores


def plot_heatmap(z_scores: pd.DataFrame, panel: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.9)
    feature_cell_types = panel["cell_type_l1"].tolist()
    fig = plt.figure(figsize=(5, 3))
    grid = GridSpec(4, 2, height_ratios=[0.22, 3.1, 1.05, 0.42], width_ratios=[1.45, 1], hspace=0.08, wspace=0.22, figure=fig)
    strip_ax = fig.add_subplot(grid[0, :])
    heatmap_ax = fig.add_subplot(grid[1, :])
    legend_ax = fig.add_subplot(grid[3, 0])
    colorbar_ax = fig.add_subplot(grid[3, 1])

    for index, cell_type in enumerate(feature_cell_types):
        strip_ax.add_patch(Rectangle((index, 0), 1, 1, facecolor=CELL_TYPE_COLORS[cell_type], edgecolor="white", linewidth=1))
    strip_ax.set_xlim(0, len(feature_cell_types))
    strip_ax.set_ylim(0, 1)
    strip_ax.axis("off")

    sns.heatmap(
        z_scores, cmap="vlag", center=0, vmin=-1.4, vmax=1.4, linewidths=0.5,
        cbar_ax=colorbar_ax, cbar_kws={"orientation": "horizontal"}, ax=heatmap_ax,
    )
    heatmap_ax.set_xlabel("")
    heatmap_ax.set_ylabel("")
    heatmap_ax.set_xticklabels(panel["feature_label"], rotation=30, ha="right", fontsize=9)
    heatmap_ax.tick_params(axis="x", pad=3)
    heatmap_ax.set_yticklabels(heatmap_ax.get_yticklabels(), rotation=0, fontsize=10)
    colorbar_ax.set_xlabel("z-score", fontsize=8, labelpad=2)
    colorbar_ax.tick_params(axis="x", labelsize=9, length=3)

    legend_ax.axis("off")
    legend_labels = {"Macro / DC": "Myeloid", "Epithelial (malig.)": "Malignant"}
    handles = [
        Patch(facecolor=color, edgecolor="none", label=legend_labels.get(cell_type, cell_type))
        for cell_type, color in CELL_TYPE_COLORS.items()
    ]
    legend_ax.legend(handles=handles, title="Cell type", loc="center", bbox_to_anchor=(0.5, 0), ncol=2,
                     frameon=False, fontsize=8, title_fontsize=8, handlelength=1.3, columnspacing=0.8)

    fig.subplots_adjust(left=0.08, right=0.985, top=0.92, bottom=-0.2)
    position = colorbar_ax.get_position()
    colorbar_ax.set_position([position.x0 + position.width * 0.05, position.y0 + position.height * 0.20,
                              position.width * 0.6, position.height * 0.5])
    fig.savefig(output_dir / "selected_signature_four_group_heatmap.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "selected_signature_four_group_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.min_cells < 1 or args.min_genes < 1:
        raise ValueError("--min-cells and --min-genes must be positive integers")
    definitions = pd.read_csv(args.signature_definitions)
    panel = pd.read_csv(args.figure_panel)
    required_definitions = {"cell_type_l1", "signature", "genes"}
    required_panel = {"cell_type_l1", "signature", "feature_label"}
    if not required_definitions.issubset(definitions.columns):
        raise ValueError("Signature definitions must contain cell_type_l1, signature, and genes")
    if not required_panel.issubset(panel.columns):
        raise ValueError("Figure panel must contain cell_type_l1, signature, and feature_label")
    defined = set(zip(definitions["cell_type_l1"], definitions["signature"]))
    missing_panel = sorted(set(zip(panel["cell_type_l1"], panel["signature"])) - defined)
    if missing_panel:
        raise ValueError(f"Figure-panel entries are absent from the full definitions: {missing_panel}")
    selected_genes = sorted({gene for genes in definitions["genes"] for gene in str(genes).split(";")})
    files = sorted(args.data_dir.glob("*.ICB_Chemo.h5ad"))
    if not files:
        raise FileNotFoundError(f"No *.ICB_Chemo.h5ad files found in {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gene_means = calculate_patient_gene_means(files, args.layer, args.min_cells, selected_genes)
    scores, coverage = calculate_signature_scores(gene_means, definitions, args.min_genes)
    group_means, z_scores = make_heatmap_matrix(scores, panel)

    group_means.to_csv(args.output_dir / "signature_four_group_mean_expression.csv")
    z_scores.to_csv(args.output_dir / "signature_four_group_heatmap_zscore.csv")
    coverage.to_csv(args.output_dir / "signature_gene_coverage.csv", index=False)
    scores.to_csv(args.output_dir / "all_patient_signature_scores.csv", index=False)
    all_group_means = scores.groupby(["cell_type_l1", "signature", "group"], observed=True)["score"].mean().unstack()
    all_group_means.to_csv(args.output_dir / "all_signature_four_group_mean_expression.csv")
    definitions.to_csv(args.output_dir / "all_signature_definitions.csv", index=False)
    panel.to_csv(args.output_dir / "signature_heatmap_features.csv", index=False)
    plot_heatmap(z_scores, panel, args.output_dir)
    print(f"Processed {len(files)} AnnData files; outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
