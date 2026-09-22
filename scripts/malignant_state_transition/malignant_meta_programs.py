#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_gene_nmf_malignant")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache_gene_nmf_malignant")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run malignant-cell multi-sample geneNMF analysis.")
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-notebook", type=Path)
    parser.add_argument("--batch-key", default="batch")
    parser.add_argument("--n-hvg", type=int, default=3000)
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=9)
    parser.add_argument("--n-meta-programs", type=int, default=10)
    parser.add_argument("--min-cells-per-sample", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument("--min-mean-similarity", type=float, default=0.25)
    parser.add_argument("--min-number-genes", type=int, default=11)
    parser.add_argument("--min-sample-coverage", type=float, default=0.30)
    return parser.parse_args()


def reference_genes_from_notebook(path: Path) -> dict[str, list[str]]:
    notebook = json.loads(path.read_text())
    result = {}
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            if output.get("output_type") != "stream":
                continue
            text = "".join(output.get("text", []))
            for line in text.splitlines():
                match = re.match(r"^(MP\d+)\s*:\s*(.*)$", line.strip())
                if match:
                    result[match.group(1)] = [gene.strip() for gene in match.group(2).split(",") if gene.strip()]
    return result


def main() -> None:
    args = parse_args()
    if args.k_min < 2 or args.k_max < args.k_min:
        raise ValueError("Invalid NMF k range")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import gene_nmf

    adata = sc.read_h5ad(args.h5ad)
    if args.batch_key not in adata.obs:
        raise ValueError(f"AnnData is missing batch column: {args.batch_key}")
    input_cells, input_genes = adata.shape
    pattern = re.compile(r"(\.)|(-AS.*)|(LINC)|(^(RPS|RPL))|(^MT-)")
    blocked = [gene for gene in adata.var_names if pattern.search(str(gene))]
    retained = [gene for gene in adata.var_names if gene not in set(blocked)]
    adata = adata[:, retained].copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg)
    hvg = adata.var_names[adata.var["highly_variable"]].tolist()
    batches = adata.obs[args.batch_key].drop_duplicates().tolist()
    adatas = [adata[adata.obs[args.batch_key].eq(batch)].copy() for batch in batches]
    batch_counts = adata.obs[args.batch_key].value_counts().reindex(batches)
    batch_table = batch_counts.rename_axis("batch").rename("n_cells").reset_index()
    batch_table["used_for_nmf"] = batch_table["n_cells"].gt(args.min_cells_per_sample)
    batch_table.to_csv(args.output_dir / "batch_cell_counts.csv", index=False)
    pd.Series(blocked, name="gene").to_csv(args.output_dir / "blocked_genes.csv", index=False)
    pd.Series(hvg, name="gene").to_csv(args.output_dir / "highly_variable_genes.csv", index=False)

    multi = gene_nmf.multi_nmf(
        adatas,
        k=range(args.k_min, args.k_max + 1),
        nmf_layer=None,
        hvg=hvg,
        nfeatures=args.n_hvg,
        min_cells_per_sample=args.min_cells_per_sample,
        random_state=args.random_state,
    )
    fit_diagnostics = pd.DataFrame([
        {"run": run, "n_iter": program.n_iter, "reached_max_iter": program.n_iter == 200}
        for run, program in multi.programs.items()
    ])
    fit_diagnostics.to_csv(args.output_dir / "nmf_fit_diagnostics.csv", index=False)
    meta = gene_nmf.get_meta_programs(multi, nMP=args.n_meta_programs)
    metrics_all = meta.metaprograms_metrics.copy()
    metrics_all.index.name = "meta_program"
    metrics_all.to_csv(args.output_dir / "meta_program_metrics_all.csv")
    drop_mask = (
        metrics_all["meanSimilarity"].lt(args.min_mean_similarity)
        | metrics_all["numberGenes"].lt(args.min_number_genes)
        | metrics_all["sampleCoverage"].lt(args.min_sample_coverage)
    )
    dropped = metrics_all.index[drop_mask].tolist()
    filtered = gene_nmf.drop_meta_programs(meta, dropped)
    metrics = filtered.metaprograms_metrics.copy()
    metrics.index.name = "meta_program"
    metrics.to_csv(args.output_dir / "meta_program_metrics_filtered.csv")
    pd.DataFrame({"meta_program": metrics_all.index, "dropped": drop_mask.to_numpy()}).to_csv(
        args.output_dir / "meta_program_filter_audit.csv", index=False)

    gene_rows = []
    for meta_program, genes in filtered.metaprograms_genes.items():
        for rank, gene in enumerate(genes, start=1):
            gene_rows.append({"meta_program": meta_program, "gene": gene, "rank": rank})
    gene_table = pd.DataFrame(gene_rows)
    gene_table.to_csv(args.output_dir / "meta_program_genes.csv", index=False)
    with (args.output_dir / "meta_program_genes.txt").open("w") as handle:
        for meta_program, genes in filtered.metaprograms_genes.items():
            handle.write(f"{meta_program} : {', '.join(genes)}\n")

    grid = gene_nmf.plot_meta_programs(
        filtered,
        similarity_cutoff=(0.15, 0.85),
        show_rownames=False,
        show_colnames=False,
        cmap="Reds",
        downsample=None,
        dendrogram_ratio=0.01,
        figsize=(5, 5),
    )
    grid.fig.savefig(args.output_dir / "meta_program_similarity.png", dpi=300, bbox_inches="tight")
    grid.fig.savefig(args.output_dir / "meta_program_similarity.pdf", bbox_inches="tight")
    plt.close(grid.fig)

    parameters = {
        "input_h5ad": args.h5ad.name,
        "input_cells": input_cells,
        "input_genes": input_genes,
        "retained_genes_after_blocklist": adata.n_vars,
        "blocked_genes": len(blocked),
        "highly_variable_genes": len(hvg),
        "batches": len(batches),
        "batches_used_for_nmf": int(batch_table["used_for_nmf"].sum()),
        "k_values": list(range(args.k_min, args.k_max + 1)),
        "nmf_runs": len(multi.programs),
        "initial_meta_programs": args.n_meta_programs,
        "dropped_meta_programs": dropped,
        "retained_meta_programs": list(filtered.metaprograms_genes),
        "random_state": args.random_state,
        "filter_rule": {
            "meanSimilarity_minimum": args.min_mean_similarity,
            "numberGenes_minimum": args.min_number_genes,
            "sampleCoverage_minimum": args.min_sample_coverage,
        },
    }
    (args.output_dir / "analysis_parameters.json").write_text(json.dumps(parameters, indent=2))

    if args.reference_notebook:
        reference = reference_genes_from_notebook(args.reference_notebook)
        comparison = []
        for meta_program in sorted(set(reference) | set(filtered.metaprograms_genes)):
            observed = filtered.metaprograms_genes.get(meta_program, [])
            expected = reference.get(meta_program, [])
            comparison.append({
                "meta_program": meta_program,
                "observed_genes": len(observed),
                "reference_genes": len(expected),
                "same_ordered_genes": observed == expected,
                "overlap_genes": len(set(observed) & set(expected)),
                "only_observed": ";".join(sorted(set(observed) - set(expected))),
                "only_reference": ";".join(sorted(set(expected) - set(observed))),
            })
        pd.DataFrame(comparison).to_csv(args.output_dir / "notebook_gene_comparison.csv", index=False)

    print(f"Input: {input_cells} cells x {input_genes} genes; batches: {len(batches)}")
    print(f"NMF runs: {len(multi.programs)}; dropped: {dropped}; retained: {list(filtered.metaprograms_genes)}")
    print("Gene counts:", {key: len(value) for key, value in filtered.metaprograms_genes.items()})


if __name__ == "__main__":
    main()
