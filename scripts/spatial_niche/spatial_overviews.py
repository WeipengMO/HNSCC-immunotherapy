#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_hnscc_spatial_overviews")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


NEIGHBORHOOD_COLORS = {"anchor": "#7b3294", "adjacent": "#d73027", "intermediate": "#fdae61", "distal": "#4575b4"}
NEIGHBORHOOD_LABELS = {"anchor": "TIMP1+ TAM-high", "adjacent": "proximal", "intermediate": "intermediate", "distal": "distal"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate complete spatial overviews for all sections.")
    parser.add_argument("--spot-scores", type=Path, required=True)
    parser.add_argument("--neighborhood-scores", type=Path, required=True)
    parser.add_argument("--proximity-scores", type=Path, required=True)
    parser.add_argument("--sample-summary", type=Path, required=True)
    parser.add_argument("--score-panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def category_overview(data, sample_order, category, colors, title, output, labels=None, size=5):
    fig, axes = plt.subplots(2, 6, figsize=(16, 7.4))
    fig.subplots_adjust(left=0.008, right=0.875, bottom=0.10, top=0.90, wspace=0.01, hspace=0.02)
    for axis, sample in zip(axes.flat, sample_order):
        section = data.loc[data["sample"].eq(sample)]
        color = section[category].map(colors).fillna("#d9d9d9")
        axis.scatter(section["pxl_col"], section["pxl_row"], c=color, s=size, linewidths=0)
        axis.set_title(sample, fontsize=9)
        axis.invert_yaxis()
        axis.set_aspect("equal")
        axis.axis("off")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=7,
                      label=(labels or {}).get(label, label)) for label, color in colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.44, 0.015))
    fig.suptitle(title, x=0.44, y=0.98, fontsize=13)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def continuous_overview(data, sample_order, score, title, output, size=5):
    values = data[score].dropna().to_numpy(dtype=float)
    robust = np.nanpercentile(values, [2, 98])
    limit = max(abs(float(robust[0])), abs(float(robust[1])))
    if limit == 0:
        limit = 1.0
    fig, axes = plt.subplots(2, 6, figsize=(16, 6.8))
    fig.subplots_adjust(left=0.008, right=0.875, bottom=0.02, top=0.90, wspace=0.01, hspace=0.02)
    scatter = None
    for axis, sample in zip(axes.flat, sample_order):
        section = data.loc[data["sample"].eq(sample)]
        scatter = axis.scatter(section["pxl_col"], section["pxl_row"], c=section[score], cmap="coolwarm",
                               vmin=-limit, vmax=limit, s=size, linewidths=0)
        axis.set_title(sample, fontsize=9)
        axis.invert_yaxis()
        axis.set_aspect("equal")
        axis.axis("off")
    colorbar_axis = fig.add_axes([0.895, 0.22, 0.016, 0.56])
    colorbar = fig.colorbar(scatter, cax=colorbar_axis, label="Score")
    colorbar.ax.tick_params(labelsize=8, length=2, pad=2)
    fig.suptitle(title, y=0.975, fontsize=13)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return limit


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spot = pd.read_csv(args.spot_scores, low_memory=False)
    neighborhood = pd.read_csv(args.neighborhood_scores, low_memory=False)
    proximity = pd.read_csv(args.proximity_scores, low_memory=False)
    summary = pd.read_csv(args.sample_summary)
    panel = pd.read_csv(args.score_panel)
    sample_order = summary["sample"].astype(str).tolist()
    if len(sample_order) != 12:
        raise ValueError(f"Expected 12 sections for the complete overview; found {len(sample_order)}")
    for frame in [spot, neighborhood, proximity]:
        missing = sorted(set(sample_order) - set(frame["sample"].astype(str)))
        if missing:
            raise ValueError(f"Input table is missing sections: {missing}")
    spot["TIMP1_TAM_high"] = spot["TIMP1_TAM_high"].astype(str).str.lower().eq("true")

    category_overview(spot, sample_order, "TIMP1_TAM_high", {True: "#d62728", False: "#d9d9d9"},
                      "TIMP1+ TAM-high spots", args.output_dir / "timp1_tam_high_overview.png")
    category_overview(neighborhood, sample_order, "tam_neighborhood", NEIGHBORHOOD_COLORS,
                      "TIMP1+ TAM spatial neighborhood groups", args.output_dir / "neighborhood_overview.png",
                      labels=NEIGHBORHOOD_LABELS)
    category_overview(proximity, sample_order, "malignant_proximity_group",
                      {"adjacent": "#d62728", "intermediate": "#fdae61", "distal": "#4575b4"},
                      "Inferred malignant spots by distance to TIMP1+ TAM-high spots",
                      args.output_dir / "malignant_proximity_overview.png", labels={"adjacent": "proximal"}, size=4)

    missing_scores = sorted(set(panel["score"]) - set(spot.columns))
    if missing_scores:
        raise ValueError(f"Configured overview scores are unavailable: {missing_scores}")
    scale_rows = []
    for row in panel.itertuples(index=False):
        limit = continuous_overview(spot, sample_order, row.score, row.title,
                                    args.output_dir / f"{row.output_stem}.png")
        scale_rows.append({"score": row.score, "vmin": -limit, "vmax": limit, "scale_rule": "symmetric global 2nd/98th percentile"})
    panel.to_csv(args.output_dir / "spatial_overview_score_panel_used.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(args.output_dir / "spatial_overview_color_scales.csv", index=False)
    print(f"Generated complete overviews for {len(sample_order)} sections, 3 categories, and {len(panel)} continuous scores")


if __name__ == "__main__":
    main()
