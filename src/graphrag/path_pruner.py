"""Relevance + diversity pruning for GraphRAG paths (K-Paths style).

Following K-Paths (Abdullahi et al., 2025), which demonstrated 90% graph size
reduction with maintained performance in cold-start settings.

Pruning jointly maximizes:
  - Relevance: cosine similarity between serialized path embedding and
    mean-pooled phenotype set description.
  - Diversity: greedy selection covering distinct proteins and relation types,
    penalizing redundant routes through high-degree hub proteins.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.graphrag.subgraph_extractor import KGPath


def compute_path_relevance(
    path_embeddings: np.ndarray,
    phenotype_query: np.ndarray,
) -> np.ndarray:
    """Cosine similarity between each path embedding and the phenotype query.

    Args:
        path_embeddings: (num_paths, embed_dim) serialized path embeddings.
        phenotype_query: (embed_dim,) mean-pooled phenotype set description embedding.

    Returns:
        Relevance scores of shape (num_paths,).
    """
    path_norms = np.linalg.norm(path_embeddings, axis=1, keepdims=True)
    path_norms = np.maximum(path_norms, 1e-8)
    query_norm = np.maximum(np.linalg.norm(phenotype_query), 1e-8)
    return (path_embeddings @ phenotype_query) / (path_norms.squeeze() * query_norm)


def greedy_diverse_selection(
    paths: list[KGPath],
    relevance_scores: np.ndarray,
    max_paths_per_drug: int = 10,
    diversity_penalty: float = 0.3,
) -> list[KGPath]:
    """Greedily select paths maximizing relevance while penalizing redundancy.

    Penalizes paths that reuse intermediate proteins already covered by
    previously selected paths. This prevents high-degree hub proteins from
    dominating the explanation.

    Args:
        paths: Candidate paths from subgraph extraction.
        relevance_scores: (num_paths,) relevance scores.
        max_paths_per_drug: Maximum paths to keep per candidate drug.
        diversity_penalty: Penalty factor for reusing covered intermediates.

    Returns:
        Pruned list of KGPath objects.
    """
    # Group paths by drug
    drug_to_paths: dict[int, list[tuple[int, KGPath]]] = defaultdict(list)
    for i, path in enumerate(paths):
        drug_to_paths[path.drug_idx].append((i, path))

    selected: list[KGPath] = []

    for drug_idx, drug_paths in drug_to_paths.items():
        if not drug_paths:
            continue

        covered_intermediates: set[int] = set()
        drug_selected: list[KGPath] = []

        # Sort by relevance descending
        drug_paths_sorted = sorted(drug_paths, key=lambda x: -relevance_scores[x[0]])

        for idx, path in drug_paths_sorted:
            if len(drug_selected) >= max_paths_per_drug:
                break

            # Compute diversity-adjusted score
            intermediates = set(path.intermediate_nodes)
            overlap = len(intermediates & covered_intermediates)
            adjusted_score = relevance_scores[idx] - diversity_penalty * overlap

            if adjusted_score > 0 or len(drug_selected) == 0:
                drug_selected.append(path)
                covered_intermediates.update(intermediates)

        selected.extend(drug_selected)

    return selected


def compute_phenotype_coverage(
    paths: list[KGPath],
    phenotype_indices: list[int],
) -> dict[int, int]:
    """Count how many patient phenotypes each drug covers across both path types.

    This coverage count is provided to the LLM as a structural prior:
    drugs addressing multiple phenotypic dimensions are more therapeutically
    compelling.

    Args:
        paths: Pruned paths.
        phenotype_indices: Patient's phenotype node indices.

    Returns:
        Dict mapping drug_idx -> number of distinct phenotypes covered.
    """
    drug_pheno_coverage: dict[int, set[int]] = defaultdict(set)
    pheno_set = set(phenotype_indices)

    for path in paths:
        if path.phenotype_idx in pheno_set:
            drug_pheno_coverage[path.drug_idx].add(path.phenotype_idx)

    return {drug: len(phenos) for drug, phenos in drug_pheno_coverage.items()}
