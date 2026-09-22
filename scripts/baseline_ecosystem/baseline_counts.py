from __future__ import annotations

import itertools
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def load_baseline_cells(h5ad_dir: Path, pattern: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(h5ad_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No h5ad files match {h5ad_dir / pattern}")
    frames = []
    for path in paths:
        data = ad.read_h5ad(path, backed="r")
        required = {"patient", "treatment", "response", "cell_type_l1", "cell_type_l3"}
        missing = sorted(required - set(data.obs.columns))
        if missing:
            data.file.close()
            raise ValueError(f"{path.name} is missing columns: {missing}")
        obs = data.obs[list(required)].copy()
        for column in obs.columns:
            obs[column] = obs[column].astype("string")
        obs = obs.loc[obs["treatment"].eq("pre") & obs["response"].isin(["R", "NR"])].copy()
        obs["source_h5ad"] = path.name
        frames.append(obs.reset_index(drop=True))
        data.file.close()
    cells = pd.concat(frames, ignore_index=True)
    dominant = (cells.groupby(["cell_type_l3", "cell_type_l1"], observed=True).size().rename("n_cells")
                .reset_index().sort_values("n_cells")
                .groupby("cell_type_l3", observed=True).tail(1)
                .set_index("cell_type_l3")["cell_type_l1"])
    cells["dominant_l1"] = cells["cell_type_l3"].map(dominant)
    contamination = (cells.loc[~cells["cell_type_l1"].eq(cells["dominant_l1"])]
                     .groupby(["cell_type_l3", "cell_type_l1"], observed=True).size()
                     .rename("n_cells").reset_index())
    cells = cells.loc[cells["cell_type_l1"].eq(cells["dominant_l1"])].drop(columns="dominant_l1")
    if cells.groupby("patient", observed=True)["response"].nunique().max() != 1:
        raise ValueError("At least one patient has multiple response labels")
    return cells, contamination


def patient_counts(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = cells[["patient", "response"]].drop_duplicates().set_index("patient")
    counts = cells.groupby(["patient", "cell_type_l1"], observed=True).size().unstack(fill_value=0)
    counts = counts.reindex(meta.index, fill_value=0)
    totals = cells.groupby("patient", observed=True).size().reindex(meta.index)
    wide = meta.join(counts).copy()
    wide["total_cells"] = totals
    long = counts.div(totals, axis=0).mul(100).stack().rename("fraction_pct").reset_index()
    long = long.merge(meta.reset_index(), on="patient", how="left", validate="many_to_one")
    return wide.reset_index(), long


def exact_group_test(values, response) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    response = np.asarray(response)
    n_nr = int((response == "NR").sum())
    observed = values[response == "NR"].mean() - values[response == "R"].mean()
    null = []
    for indices in itertools.combinations(range(len(values)), n_nr):
        mask = np.zeros(len(values), dtype=bool)
        mask[list(indices)] = True
        null.append(values[mask].mean() - values[~mask].mean())
    null = np.asarray(null)
    p_value = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    nr, responder = values[response == "NR"], values[response == "R"]
    auc = np.mean([a > b for a in nr for b in responder]) + 0.5 * np.mean([a == b for a in nr for b in responder])
    return {"NR_minus_R": observed, "rank_biserial_NR_high": 2 * auc - 1,
            "exact_permutation_p": p_value, "n_permutations": len(null),
            "n_R": int((response == "R").sum()), "n_NR": n_nr,
            "mean_R": float(values[response == "R"].mean()),
            "mean_NR": float(values[response == "NR"].mean())}


def bh_adjust(p_values) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1)
    return output
