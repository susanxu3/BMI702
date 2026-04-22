"""Inference-time degree-based score correction (Phase 2b).

High-degree drugs (e.g., Stanolone, Ribavirin) are systematically over-ranked.
This module subtracts a log-degree bias term and calibrates on validation MRR.

Can be applied independently to any scorer: base R-GCN, late fusion, or
feature-level fusion outputs.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import reciprocal_rank


def debias_scores(
    scores: np.ndarray,
    drug_degrees: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Apply degree debiasing to drug scores.

    s_debiased(d) = s_raw(d) - beta * log(deg(d) + 1)

    Args:
        scores: (num_drugs,) raw model scores.
        drug_degrees: (num_drugs,) training-set degree for each drug.
        beta: Debiasing strength (calibrated on validation MRR).

    Returns:
        Debiased scores of shape (num_drugs,).
    """
    return scores - beta * np.log(drug_degrees + 1)


def calibrate_beta(
    all_scores: dict[int, np.ndarray],
    drug_degrees: np.ndarray,
    disease_to_true_drugs: dict[int, list[int]],
    drug_indices_arr: np.ndarray,
    beta_candidates: list[float] | None = None,
) -> tuple[float, float]:
    """Find beta that maximizes macro-averaged MRR on validation diseases.

    Args:
        all_scores: Mapping from disease_idx to raw drug scores array.
        drug_degrees: (num_drugs,) degree for each drug (aligned with drug_indices_arr).
        disease_to_true_drugs: Disease -> list of ground-truth drug indices.
        drug_indices_arr: Sorted array of all drug node indices.
        beta_candidates: List of beta values to search over.

    Returns:
        (best_beta, best_mrr) tuple.
    """
    if beta_candidates is None:
        beta_candidates = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    best_beta = 0.0
    best_mrr = -1.0

    for beta in beta_candidates:
        mrrs = []
        for disease_idx, scores in all_scores.items():
            true_drugs = disease_to_true_drugs.get(disease_idx, [])
            if not true_drugs:
                continue
            debiased = debias_scores(scores, drug_degrees, beta)
            ranked_order = np.argsort(-debiased)
            ranked_drugs = drug_indices_arr[ranked_order].tolist()
            mrrs.append(reciprocal_rank(ranked_drugs, true_drugs))

        mean_mrr = float(np.mean(mrrs)) if mrrs else 0.0
        if mean_mrr > best_mrr:
            best_mrr = mean_mrr
            best_beta = beta

    return best_beta, best_mrr
