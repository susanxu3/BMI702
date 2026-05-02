"""PrimeKG graph construction and edge masking for leakage-free training.

Loads nodes.csv, edges.csv, kg.csv from the data/primekg/ directory and builds
a PyG-compatible edge_index + edge_type representation. Handles removal of all
edges incident to test disease nodes before training.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Column names in kg.csv
SRC_COL = "x_index"
DST_COL = "y_index"
REL_COL = "relation"

# Relation and node type constants
INDICATION_REL = "indication"
PHENOTYPE_REL = "disease_phenotype_positive"
OFF_LABEL_REL = "off-label use"
CONTRAINDICATION_REL = "contraindication"

DRUG_TYPE = "drug"
DISEASE_TYPE = "disease"
PHENOTYPE_TYPE = "effect/phenotype"


def load_primekg(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load PrimeKG nodes, edges, and kg DataFrames.

    Args:
        data_dir: Path to directory containing nodes.csv, edges.csv, kg.csv.

    Returns:
        (nodes_df, edges_df, kg_df) tuple of DataFrames.
    """
    data_dir = Path(data_dir)
    nodes = pd.read_csv(data_dir / "nodes.csv")
    edges = pd.read_csv(data_dir / "edges.csv")
    kg = pd.read_csv(data_dir / "kg.csv")
    return nodes, edges, kg


def mask_test_diseases(kg: pd.DataFrame, test_diseases: set[int]) -> pd.DataFrame:
    """Remove ALL edges incident to test disease nodes.

    Args:
        kg: Full PrimeKG knowledge graph DataFrame.
        test_diseases: Set of test disease node indices to mask.

    Returns:
        Filtered kg DataFrame with no test disease edges.
    """
    return kg[
        (~kg[SRC_COL].isin(test_diseases)) & (~kg[DST_COL].isin(test_diseases))
    ].copy()


def build_pyg_graph(
    kg_train: pd.DataFrame, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, int]]:
    """Build PyG edge_index and edge_type from a filtered kg DataFrame.

    Adds reverse edges for undirected message passing.

    Args:
        kg_train: Filtered knowledge graph (test diseases already removed).
        device: Target device for tensors.

    Returns:
        edge_index: (2, num_edges) LongTensor (includes reverse edges).
        edge_type: (num_edges,) LongTensor of relation types.
        num_relations: Total relation types (original * 2 for reverse).
        rel2id: Mapping from relation name to integer ID.
    """
    all_relations = sorted(kg_train[REL_COL].unique().tolist())
    rel2id = {r: i for i, r in enumerate(all_relations)}
    num_orig_rels = len(rel2id)

    src = kg_train[SRC_COL].values
    dst = kg_train[DST_COL].values
    rel = np.array([rel2id[r] for r in kg_train[REL_COL].values])

    # Add reverse edges with offset relation IDs
    edge_src = np.concatenate([src, dst])
    edge_dst = np.concatenate([dst, src])
    edge_rel = np.concatenate([rel, rel + num_orig_rels])

    num_relations = num_orig_rels * 2

    edge_index = torch.tensor(np.stack([edge_src, edge_dst]), dtype=torch.long).to(device)
    edge_type = torch.tensor(edge_rel, dtype=torch.long).to(device)

    return edge_index, edge_type, num_relations, rel2id


def build_supervision_maps(
    kg: pd.DataFrame,
    nodes: pd.DataFrame,
    train_diseases: set[int],
    test_diseases: set[int],
    train_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
) -> dict:
    """Build disease->drug and disease->phenotype lookup maps.

    Args:
        kg: Full knowledge graph.
        nodes: Node metadata.
        train_diseases: Train disease node indices.
        test_diseases: Test disease node indices.
        train_pairs: Train disease-drug pairs DataFrame.
        test_pairs: Test disease-drug pairs DataFrame.

    Returns:
        Dict with keys: disease_to_phenotypes, disease_to_drugs,
        train_disease_to_drugs, test_disease_to_drugs, drug_indices.
    """
    # Disease -> phenotype mapping
    phen_edges = kg[kg[REL_COL] == PHENOTYPE_REL]
    disease_to_phenotypes: dict[int, set[int]] = defaultdict(set)
    for _, row in phen_edges.iterrows():
        disease_to_phenotypes[row[SRC_COL]].add(row[DST_COL])
        disease_to_phenotypes[row[DST_COL]].add(row[SRC_COL])

    # Disease -> drug (indication)
    disease_to_drugs: dict[int, set[int]] = defaultdict(set)
    ind_edges = kg[kg[REL_COL] == INDICATION_REL]
    for _, row in ind_edges.iterrows():
        disease_to_drugs[row[DST_COL]].add(row[SRC_COL])

    # Train/test split supervision
    train_disease_to_drugs: dict[int, set[int]] = defaultdict(set)
    for _, row in train_pairs.iterrows():
        train_disease_to_drugs[int(row["disease_id"])].add(int(row["drug_id"]))

    test_disease_to_drugs: dict[int, set[int]] = defaultdict(set)
    for _, row in test_pairs.iterrows():
        test_disease_to_drugs[int(row["disease_id"])].add(int(row["drug_id"]))

    drug_indices = set(nodes[nodes["node_type"] == DRUG_TYPE]["node_index"].tolist())

    return {
        "disease_to_phenotypes": disease_to_phenotypes,
        "disease_to_drugs": disease_to_drugs,
        "train_disease_to_drugs": train_disease_to_drugs,
        "test_disease_to_drugs": test_disease_to_drugs,
        "drug_indices": drug_indices,
    }
