#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
from textwrap import fill

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_hnscc_timp1_gsea")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse, stats
import seaborn as sns
from statsmodels.stats.multitest import multipletests


TIMP1_CLUSTER = "Macro-TIMP1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TIMP1+ TAM rank-based pathway enrichment plots.")
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--hallmark-gmt", type=Path, required=True)
    parser.add_argument("--go-bp-gmt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def read_gmt(path: Path) -> dict[str, set[str]]:
    gene_sets = {}
    with path.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                gene_sets[parts[0]] = {gene.strip() for gene in parts[2:] if gene.strip()}
    return gene_sets


def aggregate_patient_group(adata, obs: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    selected_obs = obs.loc[mask].copy()
    counts = adata[mask].layers["counts"]
    counts = counts.tocsr() if sparse.issparse(counts) else sparse.csr_matrix(np.asarray(counts))
    patients = selected_obs["patient"].astype(str).to_numpy()
    groups = selected_obs["gsea_group"].astype(str).to_numpy()
    rows, keys = [], []
    for patient in pd.unique(patients):
        for group in pd.unique(groups[patients == patient]):
            indices = np.flatnonzero((patients == patient) & (groups == group))
            rows.append(np.asarray(counts[indices].sum(axis=0)).ravel().astype(float))
            keys.append((str(patient), str(group)))
    return pd.DataFrame(
        np.vstack(rows),
        index=pd.MultiIndex.from_tuples(keys, names=["patient", "gsea_group"]),
        columns=adata.var_names.astype(str),
    )


def paired_de(counts: pd.DataFrame) -> pd.DataFrame:
    groups = counts.index.get_level_values("gsea_group").astype(str)
    patients = counts.index.get_level_values("patient").astype(str)
    shared = sorted(
        set(patients[groups == TIMP1_CLUSTER])
        & set(patients[groups == "Other_TAM"])
    )
    if len(shared) < 2:
        raise ValueError(f"At least two paired patients are required; found {len(shared)}")
    library_size = counts.sum(axis=1).replace(0, np.nan)
    log_cpm = np.log2(counts.div(library_size, axis=0) * 1_000_000 + 0.5)
    timp1 = log_cpm.loc[pd.MultiIndex.from_product([shared, [TIMP1_CLUSTER]])].copy()
    other = log_cpm.loc[pd.MultiIndex.from_product([shared, ["Other_TAM"]])].copy()
    timp1.index = shared
    other.index = shared
    delta = timp1 - other

    rows = []
    for gene in delta.columns:
        values = delta[gene].to_numpy(dtype=float)
        test = stats.ttest_1samp(values, 0.0, nan_policy="omit")
        rows.append({
            "gene": gene,
            "mean_log2cpm_a": float(timp1[gene].mean()),
            "mean_log2cpm_b": float(other[gene].mean()),
            "log2fc_a_vs_b": float(values.mean()),
            "t_stat": float(test.statistic) if np.isfinite(test.statistic) else np.nan,
            "pval": float(test.pvalue) if np.isfinite(test.pvalue) else np.nan,
            "n_shared_patients": len(shared),
            "direction_consistency": float((values > 0).mean()),
        })
    result = pd.DataFrame(rows).set_index("gene")
    finite = np.isfinite(result["pval"].to_numpy(dtype=float))
    fdr = np.full(len(result), np.nan)
    fdr[finite] = multipletests(result.loc[finite, "pval"], method="fdr_bh")[1]
    result["fdr"] = fdr
    return result.sort_values(["fdr", "pval", "log2fc_a_vs_b"], ascending=[True, True, False], na_position="last")


def running_enrichment_score(scores: np.ndarray, hit_positions: np.ndarray):
    hit = np.zeros(len(scores), dtype=bool)
    hit[hit_positions] = True
    hit_weights = np.abs(scores[hit_positions])
    increments = np.where(hit, 0.0, -1.0 / max(1, len(scores) - len(hit_positions)))
    increments[hit_positions] = hit_weights / max(hit_weights.sum(), 1e-12)
    running = np.cumsum(increments)
    maximum = int(np.argmax(running))
    minimum = int(np.argmin(running))
    if abs(running[maximum]) >= abs(running[minimum]):
        return float(running[maximum]), maximum
    return float(running[minimum]), minimum


def preranked_gsea(de: pd.DataFrame, gene_sets: dict[str, set[str]]) -> pd.DataFrame:
    ranked = de.dropna(subset=["t_stat"])
    ranked = ranked.loc[~ranked.index.duplicated()].copy()
    ranked.index = ranked.index.astype(str)
    ranked = ranked.sort_values("t_stat", ascending=False)
    genes = ranked.index.to_numpy(dtype=str)
    scores = ranked["t_stat"].to_numpy(dtype=float)
    gene_positions = {gene: index for index, gene in enumerate(genes)}
    universe = set(genes)
    rows = []

    for term, members in gene_sets.items():
        overlap = sorted(set(map(str, members)) & universe, key=gene_positions.get)
        size = len(overlap)
        if size < 10 or size > 500:
            continue
        positions = np.array([gene_positions[gene] for gene in overlap], dtype=int)
        enrichment_score, extremum = running_enrichment_score(scores, positions)
        high_first_ranks = len(genes) - positions.astype(float)
        expected = size * (len(genes) + 1) / 2.0
        variance = size * (len(genes) - size) * (len(genes) + 1) / 12.0
        rank_z = (high_first_ranks.sum() - expected) / max(np.sqrt(variance), 1e-12)
        p_value = float(2 * stats.norm.sf(abs(rank_z)))
        if enrichment_score >= 0:
            leading = [gene for gene in overlap if gene_positions[gene] <= extremum]
        else:
            leading = [gene for gene in overlap if gene_positions[gene] >= extremum]
        rows.append({
            "term": term,
            "gene_set_size": size,
            "ES": enrichment_score,
            "rank_z": float(rank_z),
            "pval": p_value,
            "leading_edge_n": len(leading),
            "leading_edge_genes": ",".join(leading),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["fdr"] = multipletests(result["pval"], method="fdr_bh")[1]
    result["-log10_fdr"] = -np.log10(result["fdr"].clip(lower=1e-300))
    result["direction"] = np.where(result["ES"] > 0, "cluster_high", "other_TAM_high")
    return result.sort_values(["fdr", "pval", "ES"], ascending=[True, True, False])


def select_pathways(gsea: pd.DataFrame, top_n: int) -> pd.DataFrame:
    selected_tables = []
    for side, direction in [("TIMP1_TAM_high", "cluster_high"), ("Other_TAM_high", "other_TAM_high")]:
        selected = (
            gsea.loc[gsea["direction"].eq(direction)]
            .sort_values(["fdr", "pval", "ES"], ascending=[True, True, False])
            .drop_duplicates("term")
            .head(top_n)
            .copy()
        )
        selected["side"] = side
        selected["score"] = -np.log10(selected["fdr"].clip(lower=1e-300))
        selected["label"] = (
            selected["term"].str.replace("HALLMARK_", "", regex=False)
            .str.replace("GO_", "", regex=False)
            .str.replace("_", " ", regex=False)
            .str.replace(r"\s*\(GO:\d+\)$", "", regex=True)
            .str.slice(0, 65)
        )
        selected_tables.append(selected)
    return pd.concat(selected_tables, ignore_index=True)


def plot_pathways(table: pd.DataFrame, side: str, title: str, filename: str, output_dir: Path) -> None:
    plot = table.loc[table["side"].eq(side)].sort_values("score", ascending=True).reset_index(drop=True)
    y_positions = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(3.5, 5))
    bars = ax.barh(y_positions, plot["score"], height=0.78, color="#b4d4df")
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("-log10 FDR")
    ax.set_title(title)
    max_score = max(float(plot["score"].max()), 1.0) if not plot.empty else 1.0
    ax.set_xlim(0, max_score * 1.12)
    if len(plot):
        ax.set_ylim(y_positions.min() - 0.65, y_positions.max() + 0.65)
    for index, (bar, (_, row)) in enumerate(zip(bars, plot.iterrows())):
        label = fill(str(row["label"]), width=40, break_long_words=False, break_on_hyphens=False)
        padding = max(float(row["score"]) * 0.035, max_score * 0.004)
        ax.text(bar.get_x() + padding, y_positions[index], label, ha="left", va="center",
                fontsize=10, color="black", linespacing=0.95, clip_on=False)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.8)
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be a positive integer")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    adata = ad.read_h5ad(args.h5ad, backed="r")
    try:
        required = {"cell_type_l3", "treatment", "patient"}
        missing = sorted(required - set(adata.obs.columns))
        if missing:
            raise ValueError(f"AnnData is missing obs columns: {', '.join(missing)}")
        if "counts" not in adata.layers:
            raise ValueError("AnnData does not contain the 'counts' layer")
        obs = adata.obs.copy()
        for column in required:
            obs[column] = obs[column].astype("string")
        obs["cluster"] = obs["cell_type_l3"]
        pre_tam = (
            obs["treatment"].eq("pre")
            & obs["cluster"].str.startswith("Macro-", na=False)
        ).fillna(False).to_numpy(dtype=bool)
        obs["gsea_group"] = np.where(obs["cluster"].eq(TIMP1_CLUSTER), TIMP1_CLUSTER, "Other_TAM")
        counts = aggregate_patient_group(adata, obs, pre_tam)
        de = paired_de(counts)
    finally:
        adata.file.close()

    de.to_csv(args.output_dir / "DE_Macro-TIMP1_vs_other_TAM_patient_blocked.csv")
    libraries = {
        "Hallmark": read_gmt(args.hallmark_gmt),
        "GO_BP": read_gmt(args.go_bp_gmt),
    }
    gsea_tables = []
    for library, gene_sets in libraries.items():
        table = preranked_gsea(de, gene_sets)
        if table.empty:
            continue
        table.insert(0, "cluster", TIMP1_CLUSTER)
        table.insert(1, "library", library)
        gsea_tables.append(table)
    if not gsea_tables:
        raise ValueError("No pathways passed the gene-set size filters")
    gsea = pd.concat(gsea_tables, ignore_index=True)
    gsea.to_csv(args.output_dir / "GSEA_full_ranked_Macro-TIMP1_vs_other_TAM.csv", index=False)

    selected = select_pathways(gsea, args.top_n)
    selected.to_csv(args.output_dir / "GSEA_barplot_TIMP1_vs_other_TAM.csv", index=False)
    plot_pathways(selected, "TIMP1_TAM_high", "Macro-TIMP1 enriched",
                  "GSEA_barplot_TIMP1_vs_other_TAM_TIMP1_TAM.png", args.output_dir)
    plot_pathways(selected, "Other_TAM_high", "Other TAM enriched",
                  "GSEA_barplot_TIMP1_vs_other_TAM_Other_TAM.png", args.output_dir)
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
