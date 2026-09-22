#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_hnscc_mif_timp1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TIMP1 expression after MIF knockdown.")
    parser.add_argument("--de-table", type=Path, required=True)
    parser.add_argument("--tpm-table", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    de = pd.read_csv(args.de_table)
    tpm = pd.read_csv(args.tpm_table, sep="\t")
    samples = pd.read_csv(args.samples, sep="\t")
    required_sample = {"sample", "group"}
    if not required_sample.issubset(samples.columns):
        raise ValueError(f"Sample table must contain: {sorted(required_sample)}")
    if "gene_symbol" not in de or "gene_symbol" not in tpm:
        raise ValueError("DE and TPM tables must contain gene_symbol")
    sample_names = samples["sample"].astype(str).tolist()
    missing_samples = sorted(set(sample_names) - set(tpm.columns))
    if missing_samples:
        raise ValueError(f"TPM table is missing samples: {missing_samples}")
    de_gene = de.loc[de["gene_symbol"].eq("TIMP1")].copy()
    tpm_gene = tpm.loc[tpm["gene_symbol"].eq("TIMP1")].copy()
    if len(de_gene) != 1 or len(tpm_gene) != 1:
        raise ValueError("Expected exactly one TIMP1 row in each input table")

    expression = tpm_gene[sample_names].T.reset_index()
    expression.columns = ["sample", "TPM"]
    expression = expression.merge(samples[["sample", "group"]], on="sample", how="left", validate="one_to_one")
    group_order = samples["group"].drop_duplicates().tolist()
    expression["group"] = pd.Categorical(expression["group"], group_order, ordered=True)
    expression.to_csv(args.output_dir / "TIMP1_expression_source_data.csv", index=False)
    de_gene.to_csv(args.output_dir / "TIMP1_differential_expression.csv", index=False)

    palette = {group_order[0]: "#4C78A8", group_order[1]: "#E45756"}
    fig, ax = plt.subplots(figsize=(2.4, 3.5))
    sns.barplot(data=expression, x="group", y="TPM", hue="group", order=group_order,
                palette=palette, errorbar=None, legend=False, ax=ax)
    sns.stripplot(data=expression, x="group", y="TPM", order=group_order,
                  color="black", size=5, jitter=0.08, ax=ax, zorder=3)
    padj = float(de_gene["padj"].iloc[0])
    y_max = float(expression["TPM"].max())
    line_y, tick = y_max * 1.10, y_max * 0.025
    ax.plot([0, 0, 1, 1], [line_y - tick, line_y, line_y, line_y - tick], color="black", lw=1)
    ax.text(0.5, line_y + tick, f"FDR = {padj:.2e}", ha="center", va="bottom", fontsize=8)
    ax.set(xlabel="", ylabel="TPM", title="THP1 TIMP1 expression", ylim=(0, y_max * 1.28))
    sns.despine(ax=ax)
    fig.savefig(args.output_dir / "TIMP1_expression.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.output_dir / "TIMP1_expression.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Plotted TIMP1 expression for {len(expression)} samples in {len(group_order)} groups")


if __name__ == "__main__":
    main()
