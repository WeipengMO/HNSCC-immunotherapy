#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_hnscc_timp1_regulons")

import anndata as ad
import loompy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from statsmodels.stats.multitest import multipletests


TIMP1_CLUSTER = "Macro-TIMP1"
TAM_CLUSTERS = [TIMP1_CLUSTER, "Macro-FCGR3A", "Macro-FOLR2", "Macro-CD163$^-$"]
MODULE_PALETTE = {"NFKB1": "#2A9D8F", "ETS2": "#E76F51", "TCF7L2": "#5E81AC"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the TIMP1+ TAM regulon figures.")
    parser.add_argument("--loom", type=Path, required=True, help="pySCENIC output loom containing RegulonsAUC and Regulons attributes.")
    parser.add_argument("--h5ad", type=Path, required=True, help="Annotated myeloid AnnData file with a counts layer.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for figures and source-data tables.")
    parser.add_argument("--log2fc-threshold", type=float, default=1.75, help="Minimum target pseudo-bulk log2FC for the network.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of regulons displayed in summary plots.")
    parser.add_argument("--network-tfs", type=Path, required=True, help="Prespecified regulon panel for the network.")
    return parser.parse_args()


def label_regulon(regulon: str) -> str:
    return str(regulon).replace("(+)", "")


def add_bh_fdr(frame: pd.DataFrame, p_column: str, output_column: str) -> None:
    p_values = pd.to_numeric(frame[p_column], errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    finite = np.isfinite(p_values)
    if finite.any():
        adjusted[finite] = multipletests(p_values[finite], method="fdr_bh")[1]
    frame[output_column] = adjusted


def load_inputs(loom_path: Path, h5ad_path: Path):
    with loompy.connect(str(loom_path), mode="r", validate=False) as dataset:
        loom_cell_ids = pd.Index(np.asarray(dataset.ca["CellID"]).astype(str), name="cell_id")
        auc_struct = np.asarray(dataset.ca["RegulonsAUC"])
        auc_all = pd.DataFrame.from_records(auc_struct, index=loom_cell_ids)
        gene_names = np.asarray(dataset.ra["Gene"]).astype(str)
        membership = np.asarray(dataset.ra["Regulons"])
        regulon_targets = {
            regulon: set(gene_names[membership[regulon] > 0])
            for regulon in auc_struct.dtype.names
        }

    adata = ad.read_h5ad(h5ad_path, backed="r")
    required_obs = {"cell_type_l3", "treatment", "response", "patient"}
    missing = sorted(required_obs - set(adata.obs.columns))
    if missing:
        adata.file.close()
        raise ValueError(f"AnnData is missing obs columns: {', '.join(missing)}")
    if "counts" not in adata.layers:
        adata.file.close()
        raise ValueError("AnnData does not contain the 'counts' layer")
    if adata.n_vars != len(gene_names):
        adata.file.close()
        raise ValueError("The h5ad and loom gene dimensions differ")

    obs_all = adata.obs.copy()
    obs_all.index = obs_all.index.astype(str)
    overlap = obs_all.index.intersection(auc_all.index)
    if overlap.empty:
        adata.file.close()
        raise ValueError("No cell IDs overlap between the loom and h5ad files")
    obs = obs_all.loc[overlap].copy()
    for column in required_obs:
        obs[column] = obs[column].astype(str)
    auc = auc_all.loc[overlap].copy()
    h5ad_positions = pd.Series(np.arange(adata.n_obs), index=adata.obs_names.astype(str)).loc[overlap].to_numpy()
    return adata, obs, auc, h5ad_positions, gene_names, regulon_targets


def select_cohort(obs: pd.DataFrame) -> pd.DataFrame:
    keep = (
        obs["treatment"].eq("pre")
        & obs["cell_type_l3"].isin(TAM_CLUSTERS)
        & obs["response"].isin(["R", "NR"])
    )
    cohort = obs.loc[keep].copy()
    cohort["group"] = np.where(cohort["cell_type_l3"].eq(TIMP1_CLUSTER), "TIMP1+ TAM", "Other TAM")
    if cohort.empty:
        raise ValueError("No treatment-naive annotated TAM cells passed the cohort filters")
    return cohort


def compare_paired_auc(auc: pd.DataFrame, cohort: pd.DataFrame):
    patient_group = cohort[["patient", "group"]]
    means = auc.loc[cohort.index].groupby(
        [patient_group["patient"], patient_group["group"]], observed=True
    ).mean()
    timp1 = means.xs("TIMP1+ TAM", level="group")
    other = means.xs("Other TAM", level="group")
    shared = sorted(set(timp1.index.astype(str)) & set(other.index.astype(str)))
    timp1 = timp1.loc[shared]
    other = other.loc[shared]

    rows = []
    for regulon in auc.columns:
        delta = (timp1[regulon] - other[regulon]).to_numpy(dtype=float)
        t_test = stats.ttest_1samp(delta, 0.0, nan_policy="omit")
        try:
            wilcoxon_p = float(stats.wilcoxon(delta, zero_method="pratt", alternative="two-sided", method="auto").pvalue)
        except ValueError:
            wilcoxon_p = np.nan
        rows.append({
            "regulon": regulon,
            "state_delta_TIMP1_vs_other": float(np.nanmean(delta)),
            "state_median_delta": float(np.nanmedian(delta)),
            "mean_a": float(timp1[regulon].mean()),
            "mean_b": float(other[regulon].mean()),
            "direction_consistency": float(np.mean(delta > 0)),
            "effect_t": float(t_test.statistic) if np.isfinite(t_test.statistic) else np.nan,
            "p_wilcoxon": wilcoxon_p,
            "p_ttest": float(t_test.pvalue) if np.isfinite(t_test.pvalue) else np.nan,
            "n_shared_patients": len(shared),
        })
    result = pd.DataFrame(rows).set_index("regulon")
    add_bh_fdr(result, "p_wilcoxon", "fdr_wilcoxon")
    result["comparison"] = "TIMP1+ TAM vs Other TAM"
    result["regulon_label"] = result.index.map(label_regulon)
    result = result.sort_values(["state_delta_TIMP1_vs_other", "fdr_wilcoxon"], ascending=[False, True])
    return result, shared


def select_state_regulons(state_result: pd.DataFrame, top_n: int) -> list[str]:
    return state_result.sort_values(
        ["state_delta_TIMP1_vs_other", "fdr_wilcoxon"], ascending=[False, True]
    ).head(top_n).index.tolist()


def plot_state_heatmap(auc: pd.DataFrame, cohort: pd.DataFrame, top_state: list[str], output_dir: Path):
    cluster_auc = (
        auc.loc[cohort.index]
        .groupby(cohort["cell_type_l3"], observed=True)[top_state]
        .mean().T.reindex(columns=TAM_CLUSTERS)
    )
    cluster_auc.index = cluster_auc.index.map(label_regulon)
    cluster_auc_z = cluster_auc.sub(cluster_auc.mean(axis=1), axis=0).div(cluster_auc.std(axis=1).replace(0, np.nan), axis=0)

    fig, ax = plt.subplots(figsize=(3.5, max(5, len(cluster_auc_z) * 0.28)))
    sns.heatmap(cluster_auc_z, cmap="vlag", center=0, linewidths=0.2, ax=ax, cbar_kws={"label": "z-score of regulon AUC"})
    ax.set_title("Regulon activity")
    ax.set_xlabel("")
    ax.set_xticks(np.arange(cluster_auc.shape[1]) + 0.5)
    ax.set_xticklabels(cluster_auc.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(cluster_auc.shape[0]) + 0.5)
    ax.set_yticklabels(cluster_auc.index, rotation=0)
    ax.tick_params(axis="both", which="major", bottom=True, left=True, length=4, width=0.8, direction="out")
    fig.tight_layout()
    fig.savefig(output_dir / "state_regulon_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    cluster_auc.to_csv(output_dir / "state_regulon_cluster_mean_auc.csv")
    cluster_auc_z.to_csv(output_dir / "state_regulon_cluster_auc_zscore.csv")


def plot_top_regulons(auc: pd.DataFrame, cohort: pd.DataFrame, top_state: list[str], state_result: pd.DataFrame, output_dir: Path):
    state_auc = auc.loc[cohort.index[cohort["group"].eq("TIMP1+ TAM")], top_state].mean().sort_values(ascending=False)
    top = state_auc.head(10).sort_values(ascending=True)
    table = pd.DataFrame({
        "regulon": top.index.map(label_regulon),
        "mean_auc": top.to_numpy(),
        "state_delta": state_result.loc[top.index, "state_delta_TIMP1_vs_other"].to_numpy(),
        "direction_consistency": state_result.loc[top.index, "direction_consistency"].to_numpy(),
    })
    fig, ax = plt.subplots(figsize=(4, 5.5))
    colors = np.where(table["state_delta"] >= 0, "#c23b72", "#6f8fb3")
    ax.barh(table["regulon"], table["mean_auc"], color=colors, alpha=0.9)
    ax.set_xlim(0, float(table["mean_auc"].max()) * 1.18)
    ax.set_title("Top regulons in TIMP1+ TAM")
    ax.set_xlabel("Mean regulon AUC")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "timp1_tam_top_regulons.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    table.sort_values("mean_auc", ascending=False).to_csv(output_dir / "top_regulons_TIMP1_TAM.csv", index=False)


def calculate_expression_deltas(adata, h5ad_positions: np.ndarray, obs: pd.DataFrame, cohort: pd.DataFrame, gene_names: np.ndarray):
    overlap_positions = obs.index.get_indexer(cohort.index)
    counts = adata[h5ad_positions[overlap_positions]].layers["counts"].tocsr()
    # Retain the notebook's pandas sparse aggregation order so that the
    # published source-data values are reproduced to floating-point precision.
    counts_frame = pd.DataFrame.sparse.from_spmatrix(counts, index=cohort.index, columns=gene_names)
    rows, keys = [], []
    for patient in sorted(cohort["patient"].astype(str).unique()):
        patient_mask = cohort["patient"].astype(str).eq(patient)
        for group in cohort.loc[patient_mask, "group"].astype(str).unique():
            indices = cohort.index[patient_mask & cohort["group"].astype(str).eq(group)]
            rows.append(np.asarray(counts_frame.loc[indices].sum(axis=0)).ravel())
            keys.append((patient, group))
    pseudobulk = pd.DataFrame(
        np.vstack(rows),
        index=pd.MultiIndex.from_tuples(keys, names=["patient", "group"]),
        columns=gene_names,
    )
    library_size = pseudobulk.sum(axis=1).replace(0, np.nan)
    log_cpm = np.log2(pseudobulk.div(library_size, axis=0) * 1_000_000 + 0.5)
    timp1 = log_cpm.xs("TIMP1+ TAM", level="group").sort_index()
    other = log_cpm.xs("Other TAM", level="group").sort_index()
    shared = sorted(set(timp1.index.astype(str)) & set(other.index.astype(str)))
    delta = timp1.loc[shared] - other.loc[shared]
    expression_delta = delta.mean(axis=0)
    expression_delta.name = "mean_delta_a_minus_b"
    return expression_delta


def build_network_edges(regulon_targets: dict[str, set[str]], expression_delta: pd.Series, state_result: pd.DataFrame, focus_tfs: list[str]):
    missing = [tf for tf in focus_tfs if tf not in regulon_targets or tf not in state_result.index]
    if missing:
        raise ValueError(f"Required regulons are unavailable: {', '.join(missing)}")
    rows = []
    for tf in focus_tfs:
        for target in sorted(set(regulon_targets[tf]) & set(expression_delta.index)):
            fold_change = float(expression_delta.loc[target])
            if fold_change > 0:
                rows.append({
                    "tf": tf,
                    "tf_label": label_regulon(tf),
                    "target_gene": target,
                    "tf_state_delta": float(state_result.loc[tf, "state_delta_TIMP1_vs_other"]),
                    "target_expression_delta": fold_change,
                })
    edges = pd.DataFrame(rows).sort_values(["tf_label", "target_expression_delta"], ascending=[True, False])
    if edges.empty:
        raise ValueError("No positive-expression TF-target edges were found")
    return edges


def resolve_target_collisions(graph: nx.DiGraph, positions: dict[str, np.ndarray], targets: list[str], widths: dict[str, float], tf_nodes: list[str]):
    reference = {gene: positions[gene].copy() for gene in targets}
    for _ in range(450):
        displacement = {gene: np.zeros(2) for gene in targets}
        for index, gene_a in enumerate(targets):
            for gene_b in targets[index + 1:]:
                delta = positions[gene_a] - positions[gene_b]
                overlap_x = (widths[gene_a] + widths[gene_b]) / 2 + 0.018 - abs(delta[0])
                overlap_y = 0.060 + 0.020 - abs(delta[1])
                if overlap_x > 0 and overlap_y > 0:
                    if overlap_x < overlap_y:
                        push = np.array([(1 if delta[0] >= 0 else -1) * overlap_x * 0.54, 0.0])
                    else:
                        push = np.array([0.0, (1 if delta[1] >= 0 else -1) * overlap_y * 0.54])
                    displacement[gene_a] += push
                    displacement[gene_b] -= push
        for gene in targets:
            for tf in tf_nodes:
                delta = positions[gene] - positions[tf]
                distance = np.linalg.norm(delta)
                if distance < 0.18:
                    direction = delta / distance if distance > 1e-8 else np.array([1.0, 0.0])
                    displacement[gene] += direction * (0.18 - distance) * 0.65
            displacement[gene] += 0.012 * (reference[gene] - positions[gene])
        max_shift = max(np.linalg.norm(value) for value in displacement.values())
        for gene in targets:
            positions[gene] += displacement[gene]
        if max_shift < 2e-4:
            break


def plot_network(edges: pd.DataFrame, threshold: float, output_dir: Path, tf_nodes: list[str]):
    graph = nx.DiGraph()
    for row in edges.itertuples(index=False):
        graph.add_edge(row.tf_label, row.target_gene, tf=row.tf_label)
    regulator_count = edges.groupby("target_gene")["tf_label"].nunique()
    shared = sorted(regulator_count[regulator_count > 1].index)
    unique = sorted(regulator_count[regulator_count == 1].index)
    targets = unique + shared
    target_fc = edges.groupby("target_gene")["target_expression_delta"].first()

    positions = nx.kamada_kawai_layout(graph.to_undirected(), weight=None, scale=1.0)
    axis_vector = positions[tf_nodes[-1]] - positions[tf_nodes[0]]
    angle = np.arctan2(axis_vector[1], axis_vector[0])
    rotation = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
    positions = {node: rotation @ coordinates for node, coordinates in positions.items()}
    original = {node: coordinates.copy() for node, coordinates in positions.items()}
    centroid = np.mean([original[tf] for tf in tf_nodes], axis=0)
    for tf in tf_nodes:
        positions[tf] = centroid + 0.62 * (original[tf] - centroid)
    for gene in targets:
        regulators = list(graph.predecessors(gene))
        old_anchor = np.mean([original[tf] for tf in regulators], axis=0)
        new_anchor = np.mean([positions[tf] for tf in regulators], axis=0)
        radius_scale = 0.41 if "ETS2" not in regulators else 0.55
        positions[gene] = new_anchor + radius_scale * (original[gene] - old_anchor)
    positions = {node: np.array([xy[0] * 1.06, xy[1] * 0.82]) for node, xy in positions.items()}
    box_widths = {gene: 0.060 + 0.0145 * len(gene) for gene in targets}
    resolve_target_collisions(graph, positions, targets, box_widths, tf_nodes)

    soft_cmap = LinearSegmentedColormap.from_list("soft_fc", ["#FFF7D6", "#FBCB78", "#F28E5B", "#D9665B"])
    plot_norm = plt.Normalize(threshold, float(edges["target_expression_delta"].max()))
    full_norm = plt.Normalize(0, float(target_fc.max()))
    target_colors = [soft_cmap(plot_norm(float(target_fc.loc[node]))) for node in targets]
    target_edges = ["#111111" if node in shared else "#888888" for node in targets]
    target_widths = [1.15 if node in shared else 0.35 for node in targets]

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    nx.draw_networkx_edges(
        graph, positions, edge_color=[MODULE_PALETTE[graph.edges[edge]["tf"]] for edge in graph.edges],
        width=[0.65 + 0.35 * full_norm(float(target_fc.loc[target])) for _, target in graph.edges],
        alpha=0.32, arrows=True, arrowsize=6, arrowstyle="-|>", connectionstyle="arc3,rad=0.08", ax=ax,
    )
    for gene, facecolor, edgecolor, linewidth in zip(targets, target_colors, target_edges, target_widths):
        x, y = positions[gene]
        width = box_widths[gene]
        ax.add_patch(FancyBboxPatch(
            (x - width / 2, y - 0.030), width, 0.060,
            boxstyle="round,pad=0.006,rounding_size=0.008", facecolor=facecolor,
            edgecolor=edgecolor, linewidth=linewidth, alpha=0.97, zorder=3,
        ))
    nx.draw_networkx_nodes(
        graph, positions, nodelist=tf_nodes, node_shape="o", node_size=[1500] * len(tf_nodes),
        node_color=[MODULE_PALETTE[node] for node in tf_nodes], edgecolors="#202020", linewidths=1.3, ax=ax,
    )
    nx.draw_networkx_labels(graph, positions, labels={gene: gene for gene in targets}, font_size=7.4,
                            font_weight="bold", font_color="#202020", ax=ax)
    nx.draw_networkx_labels(graph, positions, labels={tf: tf for tf in tf_nodes}, font_size=9.5,
                            font_weight="bold", font_color="white", ax=ax)
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=MODULE_PALETTE["NFKB1"], markeredgecolor="#202020", markersize=11, label="TF regulon"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#F27D42", markeredgecolor="#888888", markersize=8, label="FC-positive target gene"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#F27D42", markeredgecolor="#202020", markeredgewidth=1.2, markersize=8, label="Target shared by ≥2 TFs"),
    ]
    ax.legend(handles=legend, loc="lower left", frameon=False, ncol=3, fontsize=6.5, handletextpad=0.4, columnspacing=1.0)
    ax.set_title("TIMP1+ TAM regulon network", fontsize=11, fontweight="bold", pad=20)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.margins(0.09, 0.10)
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.10, top=0.91)
    colorbar_ax = fig.add_axes([0.905, 0.675, 0.018, 0.165])
    scalar = plt.cm.ScalarMappable(norm=plot_norm, cmap=soft_cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, cax=colorbar_ax, orientation="vertical")
    colorbar.set_label("Target expression log2FC", fontsize=6.5, labelpad=5)
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.ax.tick_params(labelsize=6)
    stem = "kamada_kawai_log2FC1.75_TF_target_network"
    for suffix in ["png", "pdf", "svg"]:
        kwargs = {"dpi": 320} if suffix == "png" else {}
        fig.savefig(output_dir / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be a positive integer")
    network_panel = pd.read_csv(args.network_tfs)
    if "regulon" not in network_panel.columns:
        raise ValueError("Network TF panel must contain a regulon column")
    focus_tfs = network_panel["regulon"].astype(str).tolist()
    tf_nodes = [label_regulon(tf) for tf in focus_tfs]
    missing_colors = sorted(set(tf_nodes) - set(MODULE_PALETTE))
    if missing_colors:
        raise ValueError(f"Missing network colors for TFs: {', '.join(missing_colors)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 120

    adata, obs, auc, positions, gene_names, targets = load_inputs(args.loom, args.h5ad)
    try:
        cohort = select_cohort(obs)
        state_result, shared_patients = compare_paired_auc(auc, cohort)
        top_state = select_state_regulons(state_result, args.top_n)
        state_result.to_csv(args.output_dir / "regulon_activity_TIMP1_vs_other_TAM.csv")
        plot_state_heatmap(auc, cohort, top_state, args.output_dir)
        plot_top_regulons(auc, cohort, top_state, state_result, args.output_dir)

        expression_delta = calculate_expression_deltas(adata, positions, obs, cohort, gene_names)
        expression_delta.to_csv(args.output_dir / "expression_mean_log2cpm_delta_TIMP1_vs_other_TAM.csv")
        all_edges = build_network_edges(targets, expression_delta, state_result, focus_tfs)
        all_edges.to_csv(args.output_dir / "TF_target_edges_positive_expression.csv", index=False)
        thresholds = np.arange(0, max(2.01, args.log2fc_threshold + 0.26), 0.25)
        sensitivity = []
        for threshold in thresholds:
            passing = all_edges.loc[all_edges["target_expression_delta"].ge(threshold)]
            sensitivity.append({
                "log2fc_threshold": threshold,
                "n_edges": len(passing),
                "n_target_genes": passing["target_gene"].nunique(),
                "n_shared_target_genes": int((passing.groupby("target_gene")["tf_label"].nunique() > 1).sum()),
            })
        pd.DataFrame(sensitivity).to_csv(args.output_dir / "TF_target_network_threshold_sensitivity.csv", index=False)
        edges = all_edges.loc[all_edges["target_expression_delta"].ge(args.log2fc_threshold)].copy()
        if edges.empty or set(tf_nodes) - set(edges["tf_label"]):
            raise ValueError("The plotting threshold removes all targets from at least one prespecified TF")
        edges.to_csv(args.output_dir / "TF_target_edges_plotted.csv", index=False)
        network_panel.to_csv(args.output_dir / "TF_network_panel_used.csv", index=False)
        plot_network(edges, args.log2fc_threshold, args.output_dir, tf_nodes)
    finally:
        adata.file.close()

    print(
        f"Processed {len(cohort):,} cells from {len(shared_patients)} paired patients; "
        f"outputs written to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
