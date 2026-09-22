#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import deque
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_hnscc_timp1_neighborhood")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests


RINGS = ["anchor", "ring1", "ring2", "ring3", "ring4plus"]
RING_LABELS = {"anchor": "d = 0", "ring1": "d = 1", "ring2": "d = 2", "ring3": "d = 3", "ring4plus": "d ≥ 4"}
NEIGHBORHOOD_COLORS = {"anchor": "#7b3294", "adjacent": "#d73027", "intermediate": "#fdae61", "distal": "#4575b4"}
NEIGHBORHOOD_LABELS = {"anchor": "TIMP1+ TAM-high", "adjacent": "proximal", "intermediate": "intermediate", "distal": "distal"}
HEX_OFFSETS = [(0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze pathways across TIMP1+ TAM neighborhood rings.")
    parser.add_argument("--spot-scores", type=Path, required=True)
    parser.add_argument("--sample-summary", type=Path, required=True)
    parser.add_argument("--pathway-panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def add_neighborhood_groups(scores: pd.DataFrame) -> pd.DataFrame:
    neighborhood = scores.copy()
    neighborhood["tam_neighborhood"] = "intermediate"
    neighborhood["nearest_timp1_tam_distance_spots"] = np.nan
    for _, indices in neighborhood.groupby("sample", sort=False).groups.items():
        section = neighborhood.loc[indices]
        anchors = section.loc[section["TIMP1_TAM_high"]]
        if anchors.empty:
            continue
        xy = section[["pxl_col", "pxl_row"]].to_numpy(float)
        anchor_xy = anchors[["pxl_col", "pxl_row"]].to_numpy(float)
        spacing = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
        distance = cKDTree(anchor_xy).query(xy, k=1)[0] / spacing
        neighborhood.loc[indices, "nearest_timp1_tam_distance_spots"] = distance
        labels = np.where(distance <= 1.5, "adjacent", np.where(distance >= 3.0, "distal", "intermediate"))
        labels[section["TIMP1_TAM_high"].to_numpy(dtype=bool)] = "anchor"
        neighborhood.loc[indices, "tam_neighborhood"] = labels
    return neighborhood


def add_graph_rings(neighborhood: pd.DataFrame) -> pd.DataFrame:
    result = neighborhood.copy()
    graph_hop = np.full(len(result), np.nan)
    index_positions = pd.Series(np.arange(len(result)), index=result.index)
    for _, indices in result.groupby("sample", sort=False).groups.items():
        section = result.loc[indices].copy()
        coordinates = {
            (int(row), int(column)): local_index
            for local_index, (row, column) in enumerate(zip(section["array_row"], section["array_col"]))
        }
        anchors = np.flatnonzero(section["TIMP1_TAM_high"].to_numpy(dtype=bool))
        if not len(anchors):
            continue
        distances = np.full(len(section), np.inf)
        queue = deque()
        for local_index in anchors:
            distances[local_index] = 0
            queue.append(local_index)
        while queue:
            local_index = queue.popleft()
            row = int(section.iloc[local_index]["array_row"])
            column = int(section.iloc[local_index]["array_col"])
            for row_offset, column_offset in HEX_OFFSETS:
                neighbor = coordinates.get((row + row_offset, column + column_offset))
                if neighbor is not None and np.isinf(distances[neighbor]):
                    distances[neighbor] = distances[local_index] + 1
                    queue.append(neighbor)
        graph_hop[index_positions.loc[indices].to_numpy()] = distances
    result["timp1_tam_graph_hop"] = graph_hop
    result["timp1_tam_ring"] = np.select(
        [graph_hop == 0, graph_hop == 1, graph_hop == 2, graph_hop == 3, graph_hop >= 4],
        RINGS,
        default="unassigned",
    )
    return result


def score_columns(frame: pd.DataFrame) -> list[str]:
    columns = frame.columns.tolist()
    if "timp1_tam_state" not in columns or "log_norm_MIF" not in columns:
        raise ValueError("Could not locate the biological score column range")
    start = columns.index("timp1_tam_state")
    end = columns.index("log_norm_MIF") + 1
    return [column for column in columns[start:end] if pd.api.types.is_numeric_dtype(frame[column])]


def plot_neighborhood_overview(neighborhood: pd.DataFrame, samples: list[str], output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for axis, sample in zip(axes.flat, samples):
        section = neighborhood.loc[neighborhood["sample"].eq(sample)]
        colors = section["tam_neighborhood"].map(NEIGHBORHOOD_COLORS).fillna("#d9d9d9")
        axis.scatter(section["pxl_col"], section["pxl_row"], c=colors, s=5, linewidths=0)
        axis.set_title(sample)
        axis.invert_yaxis()
        axis.set_aspect("equal")
        axis.axis("off")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=7,
                      label=NEIGHBORHOOD_LABELS[label]) for label, color in NEIGHBORHOOD_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("TIMP1+ TAM neighborhood groups", y=0.995)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(output_dir / "timp1_tam_neighborhood_overview.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "timp1_tam_neighborhood_overview.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def summarize_rings(neighborhood: pd.DataFrame, features: list[str]):
    rows = []
    for sample, section in neighborhood.groupby("sample", sort=False):
        for ring in RINGS:
            subset = section.loc[section["timp1_tam_ring"].eq(ring)]
            if subset.empty:
                continue
            row = {"sample": sample, "ring": ring, "n_spots": len(subset)}
            row.update({feature: subset[feature].median() for feature in features})
            rows.append(row)
    section_medians = pd.DataFrame(rows)
    long = section_medians.melt(id_vars=["sample", "ring", "n_spots"], value_vars=features,
                                var_name="feature", value_name="score")
    summary = long.groupby(["feature", "ring"], sort=False, observed=True)["score"].agg(
        n_sections="count", median_score="median", q25=lambda values: values.quantile(0.25),
        q75=lambda values: values.quantile(0.75),
    ).reset_index()
    summary["ring"] = pd.Categorical(summary["ring"], categories=RINGS, ordered=True)
    summary = summary.sort_values(["feature", "ring"])
    return section_medians, long, summary


def calculate_statistics(long: pd.DataFrame, features: list[str]):
    endpoint_rows, trend_rows = [], []
    x_values = np.arange(len(RINGS))
    for feature in features:
        feature_data = long.loc[long["feature"].eq(feature)]
        wide = feature_data.pivot(index="sample", columns="ring", values="score").reindex(columns=RINGS)
        paired = wide[["anchor", "ring4plus"]].dropna()
        if len(paired) >= 3:
            p_value = float(stats.wilcoxon(paired["anchor"], paired["ring4plus"], alternative="two-sided").pvalue)
            endpoint_rows.append({"feature": feature, "n_sections": len(paired),
                                  "anchor_median": paired["anchor"].median(),
                                  "outer_median": paired["ring4plus"].median(), "p_value": p_value})
        correlations = []
        for _, values in feature_data.groupby("sample", sort=False):
            values = values.set_index("ring").reindex(RINGS)["score"]
            valid = values.notna()
            if valid.sum() >= 3 and values[valid].nunique() > 1:
                correlations.append(stats.spearmanr(x_values[valid.to_numpy()], values[valid].to_numpy()).statistic)
        trend_rows.append({"feature": feature, "n_sections": len(correlations),
                           "median_spearman_ring_rho": np.median(correlations) if correlations else np.nan,
                           "n_positive_rho": int(np.sum(np.asarray(correlations) > 0)) if correlations else 0})
    endpoint = pd.DataFrame(endpoint_rows)
    if len(endpoint):
        endpoint["fdr"] = multipletests(endpoint["p_value"], method="fdr_bh")[1]
    return endpoint, pd.DataFrame(trend_rows)


def plot_ring_scores(long: pd.DataFrame, summary: pd.DataFrame, panel: pd.DataFrame, output_dir: Path) -> None:
    features = panel["feature"].tolist()
    fig, axes = plt.subplots(1, len(features), figsize=(14, 4.5), sharex=True)
    axes = np.atleast_1d(axes)
    x_values = np.arange(len(RINGS))
    for axis, row in zip(axes, panel.itertuples(index=False)):
        feature_data = long.loc[long["feature"].eq(row.feature)]
        for _, values in feature_data.groupby("sample", sort=False):
            values = values.set_index("ring").reindex(RINGS)
            axis.plot(x_values, values["score"], color="#bdbdbd", linewidth=0.7, alpha=0.45, zorder=1)
        feature_summary = summary.loc[summary["feature"].eq(row.feature)].set_index("ring").reindex(RINGS)
        axis.fill_between(x_values, feature_summary["q25"].to_numpy(), feature_summary["q75"].to_numpy(),
                          color=row.color, alpha=0.18, linewidth=0)
        axis.plot(x_values, feature_summary["median_score"], color=row.color, marker="o", linewidth=2, markersize=5, zorder=3)
        axis.set_title(row.label)
        axis.set_xticks(x_values)
        axis.set_xticklabels([RING_LABELS[ring] for ring in RINGS], fontsize=12)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Score", fontsize=12)
    fig.supxlabel("Distance to TIMP1$^+$ TAM-high spot", y=0.1)
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    fig.savefig(output_dir / "timp1_tam_ring_pathway_scores.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "timp1_tam_ring_pathway_scores.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.spot_scores, low_memory=False)
    summary = pd.read_csv(args.sample_summary)
    panel = pd.read_csv(args.pathway_panel)
    if not {"feature", "label", "color"}.issubset(panel.columns):
        raise ValueError("Pathway table must contain feature, label, and color")
    scores["TIMP1_TAM_high"] = scores["TIMP1_TAM_high"].astype(str).str.lower().eq("true")
    samples = summary["sample"].astype(str).tolist()
    if len(samples) != 12:
        raise ValueError(f"Expected 12 sections; found {len(samples)}")
    missing_samples = sorted(set(samples) - set(scores["sample"].astype(str)))
    if missing_samples:
        raise ValueError(f"Spot-score table is missing sections: {missing_samples}")
    features = score_columns(scores)
    missing_features = sorted(set(panel["feature"]) - set(features))
    if missing_features:
        raise ValueError(f"Pathway features are unavailable: {missing_features}")

    neighborhood = add_graph_rings(add_neighborhood_groups(scores))
    neighborhood.to_csv(args.output_dir / "timp1_tam_neighborhood_spot_scores.csv", index=False)
    plot_neighborhood_overview(neighborhood, samples, args.output_dir)
    section_medians, long, ring_summary = summarize_rings(neighborhood, features)
    endpoint, trends = calculate_statistics(long, features)
    section_medians.to_csv(args.output_dir / "timp1_tam_ring_section_medians.csv", index=False)
    long.to_csv(args.output_dir / "timp1_tam_ring_scores_long.csv", index=False)
    ring_summary.to_csv(args.output_dir / "timp1_tam_ring_summary.csv", index=False)
    endpoint.to_csv(args.output_dir / "timp1_tam_ring_anchor_vs_outer.csv", index=False)
    trends.to_csv(args.output_dir / "timp1_tam_ring_trends.csv", index=False)
    panel.to_csv(args.output_dir / "timp1_tam_ring_pathways_used.csv", index=False)
    plot_ring_scores(long, ring_summary, panel, args.output_dir)
    print(f"Processed {len(samples)} sections, {len(features)} complete scores, and {len(panel)} plotted pathways")


if __name__ == "__main__":
    main()
