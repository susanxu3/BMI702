"""Error analysis: cold-start drugs, phenotype sparsity, popularity bias.

Known failure modes to analyze:
  - Cold-start drugs: 345/485 test drugs seen in training, rest get MRR ~ 0
  - Phenotype sparsity: 1-3 phenotypes -> MRR 0.13; 11+ -> MRR 0.31
  - Popularity bias: High-degree drugs (Stanolone, Ribavirin) over-ranked
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def tail_drug_evaluation(
    final_df: pd.DataFrame,
    train_pairs: pd.DataFrame,
    drug_train_count: dict[int, int] | None = None,
    tail_threshold: int = 2,
) -> pd.DataFrame:
    """Evaluate only on tail (low-degree) drugs to check popularity bias.

    Args:
        final_df: Per-disease evaluation results with 'disease' column.
        train_pairs: Training disease-drug pairs.
        drug_train_count: Pre-computed drug->training count. Built if None.
        tail_threshold: Drugs with <= this many training indications.

    Returns:
        DataFrame with tail-only metrics per disease.
    """
    if drug_train_count is None:
        drug_train_count = defaultdict(int)
        for _, row in train_pairs.iterrows():
            drug_train_count[int(row["drug_id"])] += 1

    # Filter to tail drugs per disease and recompute metrics
    # TODO: Integrate with model scoring for actual re-ranking
    return pd.DataFrame()


def sparsity_analysis(
    final_df: pd.DataFrame,
    disease_to_phenotypes: dict[int, set[int]],
) -> pd.DataFrame:
    """Stratify results by number of phenotypes per disease.

    Args:
        final_df: Per-disease results with 'disease' and 'MRR' columns.
        disease_to_phenotypes: Disease -> set of phenotype indices.

    Returns:
        DataFrame with phenotype count bins and average MRR per bin.
    """
    rows = []
    for _, row in final_df.iterrows():
        d = int(row["disease"])
        n_pheno = len(disease_to_phenotypes.get(d, set()))
        rows.append({"disease": d, "n_pheno": n_pheno, "MRR": row["MRR"]})

    df = pd.DataFrame(rows)
    bins = [0, 3, 10, float("inf")]
    labels = ["1-3", "4-10", "11+"]
    df["bin"] = pd.cut(df["n_pheno"], bins=bins, labels=labels)
    return df.groupby("bin")["MRR"].agg(["mean", "std", "count"]).reset_index()
