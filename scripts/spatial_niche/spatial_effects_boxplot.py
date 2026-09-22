#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_hnscc_spatial_effects")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot section-level malignant proximity effects.")
    parser.add_argument("--proximity", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--sample-summary", type=Path, required=True)
    parser.add_argument("--figure-panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    proximity = pd.read_csv(args.proximity, low_memory=False)
    statistics = pd.read_csv(args.statistics)
    sample_summary = pd.read_csv(args.sample_summary)
    panel = pd.read_csv(args.figure_panel)
    required_proximity = {"sample", "barcode", "malignant_inferred", "malignant_proximity_group"}
    if not required_proximity.issubset(proximity.columns):
        raise ValueError(f"Proximity table is missing columns: {sorted(required_proximity - set(proximity.columns))}")
    if not {"feature", "feature_label"}.issubset(panel.columns):
        raise ValueError("Figure panel must contain feature and feature_label")
    proximity["malignant_inferred"] = proximity["malignant_inferred"].astype(str).str.lower().eq("true")
    sample_order = sample_summary["sample"].astype(str).tolist()
    data_samples = set(proximity["sample"].astype(str))
    missing_samples = sorted(set(sample_order) - data_samples)
    if missing_samples:
        raise ValueError(f"Samples in the summary are absent from proximity data: {missing_samples}")

    all_features = [feature for feature in statistics["feature"].astype(str) if feature in proximity.columns]
    section_rows = []
    malignant = proximity.loc[proximity["malignant_inferred"]]
    for sample, section in malignant.groupby("sample", sort=False):
        for feature in all_features:
            for group in ["adjacent", "distal"]:
                values = section.loc[section["malignant_proximity_group"].eq(group), feature].dropna()
                if len(values):
                    section_rows.append({"sample": sample, "feature": feature, "group": group, "score": values.median(), "n_spots": len(values)})
    all_long = pd.DataFrame(section_rows)
    all_section = all_long.pivot_table(index=["sample", "feature"], columns="group", values="score", aggfunc="first").reset_index()
    all_section["delta_adjacent_minus_distal"] = all_section["adjacent"] - all_section["distal"]
    all_section.to_csv(args.output_dir / "spatial_effects_all_section_level_data.csv", index=False)
    statistics.to_csv(args.output_dir / "spatial_effects_all_statistics.csv", index=False)

    missing_features = sorted(set(panel["feature"]) - set(all_features))
    if missing_features:
        raise ValueError(f"Figure-panel features are unavailable: {missing_features}")
    label_map = panel.set_index("feature")["feature_label"].to_dict()
    selected_features = panel["feature"].tolist()
    group_long = all_long.loc[all_long["feature"].isin(selected_features)].copy()
    group_long["feature_label"] = group_long["feature"].map(label_map)
    selected_section = all_section.loc[all_section["feature"].isin(selected_features)].copy()
    selected_section["feature_label"] = selected_section["feature"].map(label_map)
    selected_section.to_csv(args.output_dir / "spatial_effects_boxplot_section_level_data.csv", index=False)
    panel.to_csv(args.output_dir / "spatial_effects_boxplot_panel_used.csv", index=False)
    raw_columns = ["sample", "barcode", "malignant_proximity_group"] + selected_features
    malignant.loc[malignant["malignant_proximity_group"].isin(["adjacent", "distal"]), raw_columns].to_csv(
        args.output_dir / "spatial_effects_boxplot_spot_level_input.csv", index=False
    )

    feature_order = panel["feature_label"].tolist()
    palette = {"adjacent": "#d73027", "distal": "#4575b4"}
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.boxplot(data=group_long, x="feature_label", y="score", hue="group", order=feature_order,
                hue_order=["adjacent", "distal"], palette=palette, width=0.68,
                showfliers=False, linewidth=1.0, ax=ax)
    sns.stripplot(data=group_long, x="feature_label", y="score", hue="group", order=feature_order,
                  hue_order=["adjacent", "distal"], dodge=True, palette=palette, size=4.2,
                  alpha=0.72, linewidth=0.35, edgecolor="white", ax=ax)
    feature_index = {label: index for index, label in enumerate(feature_order)}
    for feature in selected_features:
        label = label_map[feature]
        for sample in sample_order:
            pair = group_long.loc[group_long["sample"].eq(sample) & group_long["feature"].eq(feature)]
            if set(pair["group"]) == {"adjacent", "distal"}:
                adjacent = pair.loc[pair["group"].eq("adjacent"), "score"].iloc[0]
                distal = pair.loc[pair["group"].eq("distal"), "score"].iloc[0]
                x_value = feature_index[label]
                ax.plot([x_value - 0.20, x_value + 0.20], [adjacent, distal], color="#999999", alpha=0.28, linewidth=0.7, zorder=1)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ["Proximal", "Distal"], frameon=False, loc="center",
              bbox_to_anchor=(0.5, -0.35), ncol=2, markerscale=1.3, columnspacing=1.0)
    ax.axhline(0, color="#333333", linewidth=0.8, zorder=0)
    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=12)
    ax.tick_params(axis="x", rotation=30)

    score_range = float(group_long["score"].max() - group_long["score"].min())
    bracket_gap = max(0.045, score_range * 0.07)
    bracket_height = max(0.018, score_range * 0.025)
    bracket_tops = []
    for index, feature in enumerate(selected_features):
        row = statistics.loc[statistics["feature"].eq(feature)]
        if row.empty:
            continue
        p_value = float(row.iloc[0]["wilcoxon_p"])
        p_label = "p<0.001" if p_value < 0.001 else f"p={p_value:.3g}"
        pair_values = group_long.loc[group_long["feature"].eq(feature), "score"]
        bracket_y = float(pair_values.max()) + bracket_gap
        label_y = bracket_y + bracket_height * 0.18
        bracket_tops.append(label_y + bracket_height * 2.8)
        ax.plot([index - 0.20, index - 0.20, index + 0.20, index + 0.20],
                [bracket_y - bracket_height, bracket_y, bracket_y, bracket_y - bracket_height],
                color="#333333", linewidth=1.0, clip_on=False)
        ax.text(index, label_y, p_label, ha="center", va="bottom", fontsize=8)
    if bracket_tops:
        ax.set_ylim(group_long["score"].min() - score_range * 0.05, max(bracket_tops) + bracket_gap)
    fig.tight_layout()
    fig.savefig(args.output_dir / "spatial_effects_boxplot.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output_dir / "spatial_effects_boxplot.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Processed {len(sample_order)} sections and {len(all_features)} complete proximity features")


if __name__ == "__main__":
    main()
