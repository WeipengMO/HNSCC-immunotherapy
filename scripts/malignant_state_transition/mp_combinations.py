#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_hnscc_mp_combinations")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache_hnscc_mp_combinations")

import matplotlib

matplotlib.use("Agg")
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot frequent malignant meta-program combinations.")
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--mp-genes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-column", default="batch")
    parser.add_argument("--layer", default="log1p_norm")
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--sensitivity-thresholds", nargs="+", type=float, default=[0.75, 0.80, 0.85])
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260916)
    return parser.parse_args()


def prepare_sets(table: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    if not {"Module", "Gene"}.issubset(table.columns):
        raise ValueError("MP table must contain Module and Gene")
    clean = table.dropna(subset=["Module", "Gene"]).copy()
    clean["Module"] = clean["Module"].astype(str).str.strip()
    clean["Gene"] = clean["Gene"].astype(str).str.strip()
    modules = sorted(clean["Module"].unique(), key=lambda x: (len(x), x))
    return {m: tuple(dict.fromkeys(clean.loc[clean["Module"].eq(m), "Gene"])) for m in modules}


def unique_sets(gene_sets: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    sets = {m: set(genes) for m, genes in gene_sets.items()}
    return {m: tuple(sorted(sets[m] - set().union(*(sets[x] for x in sets if x != m)))) for m in sets}


def score_sets(adata: ad.AnnData, gene_sets: dict[str, tuple[str, ...]], layer: str, seed: int):
    available = set(adata.var_names.astype(str))
    scores = pd.DataFrame(index=adata.obs_names)
    qc = []
    original_x = adata.X
    try:
        adata.X = adata.layers[layer]
        for module, genes in gene_sets.items():
            matched = [gene for gene in genes if gene in available]
            qc.append({"module": module, "n_input": len(genes), "n_matched": len(matched),
                       "n_missing": len(genes) - len(matched)})
            if not matched:
                raise ValueError(f"No genes from {module} were found in the h5ad")
            key = f"__score_{module}"
            sc.tl.score_genes(adata, matched, score_name=key, random_state=seed, use_raw=False)
            scores[module] = adata.obs.pop(key).to_numpy()
    finally:
        adata.X = original_x
    return scores, pd.DataFrame(qc)


def combination_counts(active: pd.DataFrame) -> pd.DataFrame:
    labels = active.apply(lambda row: "+".join(active.columns[row.to_numpy(bool)]) or "None", axis=1)
    return (labels.value_counts().rename_axis("combination").reset_index(name="n_cells")
            .assign(fraction=lambda x: x["n_cells"] / len(active)))


def main() -> None:
    args = parse_args()
    if not 0 < args.threshold <= 1 or any(not 0 < x <= 1 for x in args.sensitivity_thresholds):
        raise ValueError("All activity thresholds must be in (0, 1]")
    if args.top_n < 1:
        raise ValueError("--top-n must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(args.h5ad)
    if args.sample_column not in adata.obs:
        raise ValueError(f"Missing sample column: {args.sample_column}")
    if args.layer not in adata.layers:
        raise ValueError(f"Missing expression layer: {args.layer}")
    original = prepare_sets(pd.read_csv(args.mp_genes))
    versions = {"original": original, "unique": unique_sets(original)}
    primary_active = None
    audit_rows = []
    for version, gene_sets in versions.items():
        scores, qc = score_sets(adata, gene_sets, args.layer, args.seed)
        qc.insert(0, "gene_set_version", version)
        qc.to_csv(args.output_dir / f"mp_score_gene_qc_{version}.csv", index=False)
        scores.to_csv(args.output_dir / f"mp_scores_{version}.csv.gz", compression="gzip")
        percentiles = scores.groupby(adata.obs[args.sample_column], observed=True).rank(method="average", pct=True)
        for threshold in sorted(set(args.sensitivity_thresholds + [args.threshold])):
            active = percentiles.ge(threshold)
            combinations = combination_counts(active)
            threshold_label = f"{threshold:.2f}".replace(".", "p")
            combinations.to_csv(args.output_dir / f"mp_combinations_{version}_{threshold_label}.csv", index=False)
            audit_rows.append({"gene_set_version": version, "threshold": threshold,
                               "n_cells": len(active), "n_combinations": len(combinations),
                               "fraction_no_active": float((active.sum(axis=1) == 0).mean()),
                               "fraction_two_or_more": float((active.sum(axis=1) >= 2).mean())})
            if version == "original" and np.isclose(threshold, args.threshold):
                primary_active = active
                primary_combinations = combinations
    pd.DataFrame(audit_rows).to_csv(args.output_dir / "mp_combination_threshold_sensitivity.csv", index=False)
    if primary_active is None:
        raise RuntimeError("Primary activity matrix was not generated")
    plotted = primary_combinations.loc[primary_combinations["combination"].ne("None")].head(args.top_n).copy()
    plotted.to_csv(args.output_dir / "mp_combinations_plotted.csv", index=False)

    modules = list(primary_active.columns)
    x = np.arange(len(plotted))
    fig, (bar_ax, matrix_ax) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.7], "hspace": 0.06})
    bar_ax.bar(x, plotted["n_cells"], color="#35B779")
    bar_ax.set(ylabel="Cells", title="Most frequent MP combinations")
    for row, module in enumerate(modules):
        on = plotted["combination"].str.split("+").apply(lambda values: module in values).to_numpy()
        matrix_ax.scatter(x[~on], np.full((~on).sum(), row), s=28, color="#DDDDDD")
        matrix_ax.scatter(x[on], np.full(on.sum(), row), s=40, color="#222222")
    for column, name in enumerate(plotted["combination"]):
        rows = [i for i, module in enumerate(modules) if module in name.split("+")]
        if len(rows) > 1:
            matrix_ax.plot([column, column], [min(rows), max(rows)], color="#222222", lw=1)
    matrix_ax.set_yticks(range(len(modules)), modules)
    matrix_ax.invert_yaxis()
    matrix_ax.set_xticks(x, plotted["combination"], rotation=60, ha="right")
    fig.tight_layout()
    fig.savefig(args.output_dir / "mp_combinations_upset.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.output_dir / "mp_combinations_upset.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Scored {adata.n_obs} cells from {adata.obs[args.sample_column].nunique()} samples; plotted {len(plotted)} combinations")


if __name__ == "__main__":
    main()
