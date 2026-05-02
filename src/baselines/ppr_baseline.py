"""Personalized PageRank baseline for drug ranking from phenotype seeds.

Phenotype nodes are used as PPR seed nodes. Drug nodes are ranked by their
PPR score. This is a structure-only baseline (MRR ~ 0.019).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix


def build_adjacency(
    kg_train_src: np.ndarray,
    kg_train_dst: np.ndarray,
    num_nodes: int,
) -> csr_matrix:
    """Build row-normalized undirected adjacency matrix.

    Args:
        kg_train_src: Source node indices from training kg.
        kg_train_dst: Destination node indices from training kg.
        num_nodes: Total number of nodes.

    Returns:
        Row-normalized sparse adjacency matrix.
    """
    rows = np.concatenate([kg_train_src, kg_train_dst])
    cols = np.concatenate([kg_train_dst, kg_train_src])
    data = np.ones(len(rows), dtype=np.float32)

    A = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))

    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    D_inv = sp.diags(1.0 / row_sums)

    return D_inv @ A


def ppr_scores(
    seed_indices: list[int],
    A_norm: csr_matrix,
    alpha: float = 0.15,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    """Compute Personalized PageRank scores for a set of seed nodes.

    Args:
        seed_indices: Phenotype node indices as seeds.
        A_norm: Row-normalized adjacency (N x N).
        alpha: Restart probability.
        max_iter: Maximum power iterations.
        tol: Convergence threshold (L1 norm).

    Returns:
        PPR score for every node, shape (N,).
    """
    N = A_norm.shape[0]
    s = np.zeros(N, dtype=np.float32)
    if len(seed_indices) == 0:
        return s
    s[seed_indices] = 1.0 / len(seed_indices)

    r = s.copy()
    A_T = A_norm.T

    for _ in range(max_iter):
        r_new = (1 - alpha) * A_T.dot(r) + alpha * s
        if np.linalg.norm(r_new - r, 1) < tol:
            break
        r = r_new

    return r_new
