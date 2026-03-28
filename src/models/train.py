"""Training loop for PhenoDrugModel (R-GCN + cross-attention).

Supports:
  - Margin-based ranking loss with degree-weighted negative sampling
  - Gradient accumulation and gradient clipping
  - Early stopping on validation MRR
  - Checkpoint saving
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from src.data.primekg_loader import build_pyg_graph, load_primekg
from src.data.disease_split import load_split
from src.evaluation.metrics import recall_at_k, reciprocal_rank
from src.models.cross_attention_scorer import PhenoDrugModel
from src.models.rgcn_encoder import RGCNEncoder
from src.utils.negative_sampling import DegreeWeightedSampler


def pad_pheno_batch(
    pheno_lists: list[list[int]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length phenotype lists into a batch tensor + mask.

    Args:
        pheno_lists: List of phenotype index lists (variable length).
        device: Target device.

    Returns:
        padded: (batch, max_phenos) LongTensor of phenotype indices.
        mask: (batch, max_phenos) BoolTensor, True where padded.
    """
    max_len = max(len(p) for p in pheno_lists)
    batch_size = len(pheno_lists)
    padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)
    for i, p in enumerate(pheno_lists):
        padded[i, : len(p)] = torch.tensor(p, dtype=torch.long)
        mask[i, : len(p)] = False
    return padded, mask


def evaluate(
    model: PhenoDrugModel,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    test_diseases: set[int],
    disease_to_phenotypes: dict[int, set[int]],
    test_disease_to_drugs: dict[int, set[int]],
    drug_indices_arr: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on test diseases, returning macro-averaged metrics.

    Args:
        model: Trained PhenoDrugModel.
        edge_index: Graph edge indices.
        edge_type: Edge relation types.
        test_diseases: Set of test disease node indices.
        disease_to_phenotypes: Disease -> phenotype node indices.
        test_disease_to_drugs: Disease -> ground-truth drug node indices.
        drug_indices_arr: Sorted array of all drug node indices.
        device: Compute device.

    Returns:
        Dict with macro-averaged MRR, R@1, R@5, R@10, R@50.
    """
    model.eval()
    with torch.no_grad():
        node_embs = model.encode(edge_index, edge_type)
        results = []
        for disease_idx in test_diseases:
            phenos = list(disease_to_phenotypes.get(disease_idx, []))
            true_drugs = list(test_disease_to_drugs.get(disease_idx, []))
            if not phenos or not true_drugs:
                continue

            all_drugs_t = torch.tensor(drug_indices_arr, dtype=torch.long, device=device)
            chunk = 512
            all_scores = []
            for c_start in range(0, len(drug_indices_arr), chunk):
                c_end = min(c_start + chunk, len(drug_indices_arr))
                c_drugs = all_drugs_t[c_start:c_end]
                c_padded, c_mask = pad_pheno_batch([phenos] * (c_end - c_start), device)
                scores = model.score(node_embs, c_drugs, c_padded, c_mask)
                all_scores.append(scores.cpu().numpy())
            all_scores = np.concatenate(all_scores)

            ranked_order = np.argsort(-all_scores)
            ranked_drugs = drug_indices_arr[ranked_order].tolist()

            results.append({
                "R@1": recall_at_k(ranked_drugs, true_drugs, 1),
                "R@5": recall_at_k(ranked_drugs, true_drugs, 5),
                "R@10": recall_at_k(ranked_drugs, true_drugs, 10),
                "R@50": recall_at_k(ranked_drugs, true_drugs, 50),
                "MRR": reciprocal_rank(ranked_drugs, true_drugs),
            })

    if not results:
        return {"MRR": 0.0, "R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "R@50": 0.0}

    return {k: np.mean([r[k] for r in results]) for k in results[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PhenoDrugModel")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # TODO: Wire up config-driven training loop
    print(f"Loaded config: {cfg}")
    print("Training not yet wired — run from notebook for now.")


if __name__ == "__main__":
    main()
