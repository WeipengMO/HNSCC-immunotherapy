#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_hnscc_mif_genes")

import matplotlib

matplotlib.use("Agg")
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot representative genes for MIF-responsive pathways.")
    parser.add_argument("--de-table", type=Path, required=True)
    parser.add_argument("--vst-table", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--programs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fdr", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    de = pd.read_csv(args.de_table)
    vst = pd.read_csv(args.vst_table)
    samples = pd.read_csv(args.samples, sep="\t")
    programs = pd.read_csv(args.programs)
    required = {"pathway", "direction", "color", "source_terms", "genes"}
    if not required.issubset(programs.columns):
        raise ValueError(f"Program configuration must contain: {sorted(required)}")
    if not set(programs["direction"]).issubset({"up", "down"}):
        raise ValueError("Program direction must be up or down")
    sample_names = samples["sample"].astype(str).tolist()
    if not {"sample", "group"}.issubset(samples.columns) or not set(sample_names).issubset(vst.columns):
        raise ValueError("Sample metadata or VST sample columns are incomplete")

    de_by_gene = (
        de.dropna(subset=["gene_symbol", "stat"])
        .assign(gene_symbol=lambda x: x["gene_symbol"].astype(str).str.upper())
        .sort_values("padj", na_position="last")
        .drop_duplicates("gene_symbol")
        .set_index("gene_symbol")
    )
    expression = (
        vst.dropna(subset=["gene_symbol"])
        .assign(gene_symbol=lambda x: x["gene_symbol"].astype(str).str.upper())
        .groupby("gene_symbol")[sample_names].mean()
    )
    expression_z = expression.sub(expression.mean(axis=1), axis=0)
    expression_z = expression_z.div(expression.std(axis=1, ddof=0).replace(0, np.nan), axis=0).fillna(0)

    audit_rows = []
    seen = set()
    for program in programs.itertuples(index=False):
        genes = [gene.strip().upper() for gene in str(program.genes).split(";") if gene.strip()]
        for gene in genes:
            in_de = gene in de_by_gene.index
            in_expression = gene in expression_z.index
            row = de_by_gene.loc[gene] if in_de else None
            padj = row["padj"] if in_de else np.nan
            log2fc = row["log2FoldChange"] if in_de else np.nan
            direction_matches = bool(
                in_de and ((program.direction == "down" and log2fc < 0) or (program.direction == "up" and log2fc > 0))
            )
            selected = bool(in_de and in_expression and padj < args.fdr and direction_matches and gene not in seen)
            audit_rows.append({
                "pathway": program.pathway, "gene": gene, "color": program.color,
                "expected_direction": program.direction, "log2FoldChange": log2fc,
                "stat": row["stat"] if in_de else np.nan, "padj": padj,
                "source_terms": program.source_terms, "present_in_de": in_de,
                "present_in_vst": in_expression, "passes_fdr": bool(in_de and padj < args.fdr),
                "direction_matches": direction_matches, "not_previously_used": gene not in seen,
                "selected": selected,
            })
            if selected:
                seen.add(gene)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(args.output_dir / "representative_gene_audit.csv", index=False)
    selected = audit.loc[audit["selected"]].copy()
    selected.to_csv(args.output_dir / "representative_genes_selected.csv", index=False)
    programs.to_csv(args.output_dir / "representative_gene_programs.csv", index=False)
    if selected.empty:
        raise ValueError("No configured genes pass expression, FDR and direction checks")

    gene_order = selected["gene"].tolist()
    matrix = expression_z.loc[gene_order, sample_names].T
    matrix.index = samples.set_index("sample").loc[sample_names].apply(
        lambda row: f"{row.name} ({row['group']})", axis=1
    )
    matrix.to_csv(args.output_dir / "representative_gene_heatmap_source_data.csv")

    sns.set_theme(style="whitegrid", context="notebook")
    fig = plt.figure(figsize=(max(5, 0.25 * len(gene_order)), 5))
    outer = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[0.20, 3.5, 0.82, 0.82], hspace=0.06)
    annotation_ax = fig.add_subplot(outer[0])
    heatmap_ax = fig.add_subplot(outer[1])
    bottom = outer[3].subgridspec(1, 2, width_ratios=[3.2, 1.2], wspace=0.4)
    legend_ax = fig.add_subplot(bottom[0])
    cbar_ax = fig.add_subplot(bottom[1])

    annotation_ax.imshow(
        np.arange(len(gene_order))[None, :], aspect="auto",
        cmap=ListedColormap(selected["color"].tolist()), interpolation="nearest",
        vmin=-0.5, vmax=len(gene_order) - 0.5,
    )
    annotation_ax.set_xlim(-0.5, len(gene_order) - 0.5)
    annotation_ax.set_xticks([])
    annotation_ax.set_yticks([])
    for spine in annotation_ax.spines.values():
        spine.set_visible(False)

    vmax = max(1.5, float(np.nanmax(np.abs(matrix.to_numpy()))))
    sns.heatmap(
        matrix, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        linewidths=0.45, linecolor="white", cbar=False, ax=heatmap_ax,
    )
    heatmap_ax.set(xlabel="", ylabel="")
    heatmap_ax.set_xticks(np.arange(len(gene_order)) + 0.5)
    heatmap_ax.set_xticklabels(gene_order, rotation=30, ha="right", rotation_mode="anchor", fontsize=8)
    heatmap_ax.set_yticklabels(matrix.index.str.replace(r"\s*\(.*\)$", "", regex=True))
    heatmap_ax.tick_params(axis="x", bottom=True, labelbottom=True, pad=2)
    heatmap_ax.tick_params(axis="y", rotation=0, labelsize=10)

    legend_ax.axis("off")
    handles = [Patch(facecolor=row.color, edgecolor="none", label=row.pathway) for row in programs.itertuples()]
    legend_ax.legend(
        handles=handles, title="Pathway", frameon=False, ncol=2, loc="center left",
        bbox_to_anchor=(0, 0.5), handlelength=1.2, columnspacing=1.2, fontsize=10, title_fontsize=9.5,
    )
    cbar_ax.axis("off")
    thin_cbar_ax = cbar_ax.inset_axes([0.03, 0.38, 0.94, 0.24])
    scalar_mappable = ScalarMappable(norm=Normalize(vmin=-vmax, vmax=vmax), cmap="RdBu_r")
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(
        scalar_mappable, cax=thin_cbar_ax, orientation="horizontal",
    )
    colorbar.set_label("z-score", labelpad=3)
    colorbar.ax.tick_params(labelsize=8, length=2.5)
    fig.savefig(args.output_dir / "representative_gene_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output_dir / "representative_gene_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Selected {len(selected)} configured genes from {len(programs)} pathway groups")


if __name__ == "__main__":
    main()
