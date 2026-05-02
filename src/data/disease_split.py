"""80/20 disease-level train/test split logic.

Eligible diseases must have both >=1 phenotype AND >=1 drug indication.
Split: 431 train, 108 test (from 539 eligible diseases).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_split(
    split_dir: str | Path,
) -> tuple[set[int], set[int], pd.DataFrame, pd.DataFrame]:
    """Load pre-computed train/test split files.

    Args:
        split_dir: Path to directory with train_disease_ids.txt, test_disease_ids.txt,
            train_drug_pairs.csv, test_drug_pairs.csv.

    Returns:
        train_diseases: Set of train disease node indices.
        test_diseases: Set of test disease node indices.
        train_pairs: DataFrame of (disease_id, drug_id) training pairs.
        test_pairs: DataFrame of (disease_id, drug_id) test pairs.
    """
    split_dir = Path(split_dir)

    train_diseases = set(
        int(x.strip()) for x in (split_dir / "train_disease_ids.txt").read_text().splitlines() if x.strip()
    )
    test_diseases = set(
        int(x.strip()) for x in (split_dir / "test_disease_ids.txt").read_text().splitlines() if x.strip()
    )
    train_pairs = pd.read_csv(split_dir / "train_drug_pairs.csv")
    test_pairs = pd.read_csv(split_dir / "test_drug_pairs.csv")

    return train_diseases, test_diseases, train_pairs, test_pairs
