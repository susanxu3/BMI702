"""Per-disease ranking metrics: MRR, Recall@K, AUROC, AUPRC.

All metrics are computed per-disease, then macro-averaged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def recall_at_k(ranked_drugs: list[int], true_drugs: list[int], k: int) -> float:
    """Fraction of true drugs found in the top-k ranked list.

    Args:
        ranked_drugs: Drug indices sorted by descending score.
        true_drugs: Ground-truth drug indices.
        k: Cutoff rank.

    Returns:
        Recall at k.
    """
    if not true_drugs:
        return 0.0
    top_k = set(ranked_drugs[:k])
    hits = len(top_k & set(true_drugs))
    return hits / len(true_drugs)


def reciprocal_rank(ranked_drugs: list[int], true_drugs: list[int]) -> float:
    """Reciprocal rank of the first true drug in the ranked list.

    Args:
        ranked_drugs: Drug indices sorted by descending score.
        true_drugs: Ground-truth drug indices.

    Returns:
        1/rank of first hit, or 0.0 if no hit.
    """
    true_set = set(true_drugs)
    for rank, d in enumerate(ranked_drugs, start=1):
        if d in true_set:
            return 1.0 / rank
    return 0.0


def per_disease_auroc(
    scores: np.ndarray, true_drug_indices: set[int], all_drug_indices: np.ndarray
) -> float | None:
    """Full-library AUROC for a single disease.

    Args:
        scores: Predicted scores for all drugs (aligned with all_drug_indices).
        true_drug_indices: Set of ground-truth drug indices.
        all_drug_indices: Array of all drug node indices.

    Returns:
        AUROC or None if degenerate (all positive or all negative).
    """
    labels = np.array([1 if d in true_drug_indices else 0 for d in all_drug_indices])
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None
    return float(roc_auc_score(labels, scores))


def per_disease_auprc(
    scores: np.ndarray, true_drug_indices: set[int], all_drug_indices: np.ndarray
) -> float | None:
    """Full-library AUPRC for a single disease (imbalanced, macro-averaging ready).

    Mirrors `per_disease_auroc` — computes AP over all 7957 drugs per disease so
    the caller can macro-average across diseases. This is the "honest" per-task
    AUPRC; pair it with a TxGNN-style pooled 1:1 balanced number in the caller
    if you want comparability with the TxGNN paper.

    Args:
        scores: Predicted scores for all drugs (aligned with all_drug_indices).
        true_drug_indices: Set of ground-truth drug indices.
        all_drug_indices: Array of all drug node indices.

    Returns:
        AUPRC or None if no positives.
    """
    labels = np.array([1 if d in true_drug_indices else 0 for d in all_drug_indices])
    if labels.sum() == 0:
        return None
    return float(average_precision_score(labels, scores))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved model on test split")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    args = parser.parse_args()
    # TODO: Load model, data, run evaluation
    print(f"Evaluating {args.model} on {args.split} split")


if __name__ == "__main__":
    main()
