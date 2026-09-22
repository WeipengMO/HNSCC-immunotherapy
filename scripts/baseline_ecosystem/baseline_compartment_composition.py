#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_hnscc_baseline_composition")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns

from baseline_counts import bh_adjust, exact_group_test, load_baseline_cells, patient_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot baseline compartment composition by response.")
    parser.add_argument("--h5ad-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.ICB_Chemo.h5ad")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells, contamination = load_baseline_cells(args.h5ad_dir, args.pattern)
    counts, all_composition = patient_counts(cells)
    groups = ["T", "Macro / DC", "Fibroblast"]
    missing = sorted(set(groups) - set(all_composition["cell_type_l1"]))
    if missing:
        raise ValueError(f"Missing level-1 cell groups: {missing}")
    plotted = all_composition.loc[all_composition["cell_type_l1"].isin(groups)].copy()
    plotted["cell_group"] = plotted["cell_type_l1"].replace({"Macro / DC": "Myeloid"})
    order = ["T", "Myeloid", "Fibroblast"]
    plotted["cell_group"] = pd.Categorical(plotted["cell_group"], order, ordered=True)
    all_composition.to_csv(args.output_dir / "baseline_all_compartment_composition.csv", index=False)
    plotted.to_csv(args.output_dir / "baseline_compartment_composition_source_data.csv", index=False)
    counts.to_csv(args.output_dir / "baseline_patient_cell_counts.csv", index=False)
    contamination.to_csv(args.output_dir / "cross_parent_labels_removed.csv", index=False)

    rows = []
    for group in order:
        subset = plotted.loc[plotted["cell_group"].eq(group)]
        result = exact_group_test(subset["fraction_pct"], subset["response"])
        rows.append({"cell_group": group, **result})
    statistics = pd.DataFrame(rows)
    statistics["fdr"] = bh_adjust(statistics["exact_permutation_p"])
    statistics.to_csv(args.output_dir / "baseline_compartment_composition_statistics.csv", index=False)

    colors = {"R": "#2166AC", "NR": "#B2182B"}
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.boxplot(data=plotted, x="cell_group", y="fraction_pct", hue="response", order=order,
                hue_order=["R", "NR"], palette={"R": "#EEEEEE", "NR": "#EEEEEE"},
                width=0.65, fliersize=0, ax=ax)
    sns.stripplot(data=plotted, x="cell_group", y="fraction_pct", hue="response", order=order,
                  hue_order=["R", "NR"], palette=colors, dodge=True, jitter=0.08,
                  size=4, legend=False, ax=ax)
    value_range = max(plotted["fraction_pct"].max() - plotted["fraction_pct"].min(), 1)
    ax.set_ylim(plotted["fraction_pct"].min() - 0.04 * value_range,
                plotted["fraction_pct"].max() + 0.06 * value_range)
    ax.set(xlabel="", ylabel="Cell fraction (%)")
    ax.legend(handles=[Patch(facecolor=colors[x], label=x) for x in ["R", "NR"]],
              title="Response", frameon=False)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(args.output_dir / "baseline_compartment_composition.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.output_dir / "baseline_compartment_composition.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Analyzed {len(counts)} patients and {len(cells)} cells across {len(all_composition['cell_type_l1'].unique())} level-1 groups")


if __name__ == "__main__":
    main()
