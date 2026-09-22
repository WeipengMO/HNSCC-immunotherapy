#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_hnscc_baseline_balance")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from baseline_counts import exact_group_test, load_baseline_cells, patient_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot baseline T-cell to myeloid-fibroblast balance.")
    parser.add_argument("--h5ad-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.ICB_Chemo.h5ad")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pseudocount", type=float, default=0.5)
    return parser.parse_args()


def calculate_ratio(counts: pd.DataFrame, pseudocount: float) -> pd.Series:
    required = ["T", "Macro / DC", "Fibroblast"]
    missing = sorted(set(required) - set(counts.columns))
    if missing:
        raise ValueError(f"Missing level-1 cell groups: {missing}")
    return np.log2((counts["T"] + pseudocount) /
                   (counts["Macro / DC"] + counts["Fibroblast"] + pseudocount))


def main() -> None:
    args = parse_args()
    if args.pseudocount <= 0:
        raise ValueError("--pseudocount must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells, contamination = load_baseline_cells(args.h5ad_dir, args.pattern)
    counts, all_composition = patient_counts(cells)
    counts["log2_t_over_myeloid_fibroblast"] = calculate_ratio(counts, args.pseudocount)
    counts.to_csv(args.output_dir / "baseline_patient_cell_counts_and_balance.csv", index=False)
    all_composition.to_csv(args.output_dir / "baseline_all_compartment_composition.csv", index=False)
    contamination.to_csv(args.output_dir / "cross_parent_labels_removed.csv", index=False)

    metric = "log2_t_over_myeloid_fibroblast"
    stats = exact_group_test(counts[metric], counts["response"])
    pd.DataFrame([{"metric": metric, "pseudocount": args.pseudocount, **stats}]).to_csv(
        args.output_dir / "t_cell_compartment_balance_statistics.csv", index=False)
    leave_one_out = []
    for patient in counts["patient"]:
        subset = counts.loc[counts["patient"].ne(patient)]
        leave_one_out.append({"dropped_patient": patient, **exact_group_test(subset[metric], subset["response"])})
    pd.DataFrame(leave_one_out).to_csv(args.output_dir / "t_cell_compartment_balance_leave_one_out.csv", index=False)
    sensitivity = []
    for pseudocount in [0.1, 0.5, 1.0, 2.0]:
        values = calculate_ratio(counts, pseudocount)
        sensitivity.append({"pseudocount": pseudocount, **exact_group_test(values, counts["response"])})
    pd.DataFrame(sensitivity).to_csv(args.output_dir / "t_cell_compartment_balance_pseudocount_sensitivity.csv", index=False)

    colors = {"R": "#2166AC", "NR": "#B2182B"}
    fig, ax = plt.subplots(figsize=(2.5, 3))
    sns.boxplot(data=counts, x="response", y=metric, order=["R", "NR"], color="#EEEEEE", fliersize=0, ax=ax)
    sns.stripplot(data=counts, x="response", y=metric, order=["R", "NR"], hue="response",
                  palette=colors, size=6, legend=False, ax=ax)
    ymin, ymax = counts[metric].min(), counts[metric].max()
    value_range = max(ymax - ymin, 1)
    bar_y, tick = ymax + 0.10 * value_range, 0.035 * value_range
    ax.plot([0, 0, 1, 1], [bar_y - tick, bar_y, bar_y, bar_y - tick], color="black", lw=1, clip_on=False)
    ax.text(0.5, bar_y + 0.025 * value_range, f"p = {stats['exact_permutation_p']:.3g}", ha="center", va="bottom", fontsize=9)
    ax.set(xlabel="", ylabel=r"$\log_2$(T cell / myeloid–fibroblast ratio)",
           ylim=(ymin - 0.08 * value_range, bar_y + 0.15 * value_range))
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(args.output_dir / "t_cell_compartment_balance.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.output_dir / "t_cell_compartment_balance.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Analyzed {len(counts)} patients ({stats['n_R']} R, {stats['n_NR']} NR) from {len(cells)} cells")


if __name__ == "__main__":
    main()
