#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_hnscc_mp_overlap")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot overlap among malignant meta-program gene sets.")
    parser.add_argument("--mp-genes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.mp_genes)
    if not {"Module", "Gene"}.issubset(table.columns):
        raise ValueError("MP table must contain Module and Gene")
    table = table.dropna(subset=["Module", "Gene"]).copy()
    table["Module"] = table["Module"].astype(str).str.strip()
    table["Gene"] = table["Gene"].astype(str).str.strip()
    table = table.loc[table["Gene"].ne("")].drop_duplicates(["Module", "Gene"])
    modules = sorted(table["Module"].unique(), key=lambda x: (len(x), x))
    gene_sets = {module: set(table.loc[table["Module"].eq(module), "Gene"]) for module in modules}
    if len(modules) < 2:
        raise ValueError("At least two modules are required")

    shared = pd.DataFrame(index=modules, columns=modules, dtype=int)
    jaccard = pd.DataFrame(index=modules, columns=modules, dtype=float)
    for first in modules:
        for second in modules:
            intersection = gene_sets[first] & gene_sets[second]
            union = gene_sets[first] | gene_sets[second]
            shared.loc[first, second] = len(intersection)
            jaccard.loc[first, second] = len(intersection) / len(union)
    shared_long = []
    for first, second in combinations(modules, 2):
        for gene in sorted(gene_sets[first] & gene_sets[second]):
            shared_long.append({"module_a": first, "module_b": second, "gene": gene})
    counts = pd.DataFrame({
        "module": modules,
        "n_genes": [len(gene_sets[m]) for m in modules],
        "n_unique": [len(gene_sets[m] - set().union(*(gene_sets[x] for x in modules if x != m))) for m in modules],
    })
    table.to_csv(args.output_dir / "mp_gene_definitions.csv", index=False)
    counts.to_csv(args.output_dir / "mp_gene_counts.csv", index=False)
    shared.to_csv(args.output_dir / "mp_shared_gene_counts.csv")
    jaccard.to_csv(args.output_dir / "mp_gene_jaccard.csv")
    pd.DataFrame(shared_long, columns=["module_a", "module_b", "gene"]).to_csv(
        args.output_dir / "mp_shared_genes.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(jaccard, annot=True, fmt=".2f", cmap="Blues", square=True, ax=ax)
    ax.set(xlabel="", ylabel="", title="MP gene-set Jaccard similarity")
    fig.tight_layout()
    fig.savefig(args.output_dir / "mp_gene_overlap_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.output_dir / "mp_gene_overlap_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Compared all {len(modules)} modules containing {len(table)} distinct module-gene pairs")


if __name__ == "__main__":
    main()
