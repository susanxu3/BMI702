"""Degree-weighted negative sampling for margin-based ranking loss.

Popular drugs are sampled more often as negatives to counteract popularity bias.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


class DegreeWeightedSampler:
    """Negative drug sampler weighted by training-set drug degree.

    Args:
        train_pairs: DataFrame with 'drug_id' column.
        drug_indices: Sorted list of all drug node indices.
        seed: Random seed.
    """

    def __init__(self, train_pairs: pd.DataFrame, drug_indices: list[int], seed: int = 42) -> None:
        self.drug_list = sorted(drug_indices)
        self.rng = np.random.default_rng(seed)

        drug_degree: dict[int, int] = defaultdict(int)
        for _, row in train_pairs.iterrows():
            drug_degree[int(row["drug_id"])] += 1

        weights = np.array([drug_degree.get(d, 0) + 1 for d in self.drug_list], dtype=np.float64)
        self.weights = weights / weights.sum()

    def sample(self, positive_drug: int, disease_true_drugs: set[int], n: int = 1) -> list[int]:
        """Sample n negative drugs, avoiding positives for this disease.

        Args:
            positive_drug: The positive drug to avoid.
            disease_true_drugs: All true drugs for this disease.
            n: Number of negatives to sample.

        Returns:
            List of n negative drug indices.
        """
        negs: list[int] = []
        while len(negs) < n:
            candidates = self.rng.choice(self.drug_list, size=n * 2, p=self.weights)
            for c in candidates:
                if c not in disease_true_drugs and c != positive_drug:
                    negs.append(int(c))
                    if len(negs) == n:
                        break
        return negs
