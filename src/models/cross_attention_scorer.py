"""Drug-conditioned cross-attention scorer + flat PhenoDrugModel.

Drug embedding = query, phenotype embeddings = keys/values.
Produces a scalar relevance score per (drug, phenotype-set) pair.

PhenoDrugModel keeps a flat parameter layout (node_emb, convs, norms,
cross_attn) that matches the training notebook (`src/pagerank & new model.ipynb`),
so checkpoints saved by that notebook load directly via `load_state_dict`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class DrugConditionedCrossAttention(nn.Module):
    """Multi-head cross-attention: drug queries over phenotype keys/values.

    Args:
        dim: Embedding dimension (must match encoder output).
        num_heads: Number of attention heads.
        dropout: Attention dropout rate.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.score_proj = nn.Linear(dim, 1)

    def forward(
        self,
        drug_emb: torch.Tensor,
        pheno_embs: torch.Tensor,
        pheno_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score drugs against a phenotype set.

        Args:
            drug_emb: (batch, dim) drug embeddings.
            pheno_embs: (batch, max_phenos, dim) phenotype embeddings.
            pheno_mask: (batch, max_phenos) True where padded.

        Returns:
            Scalar scores of shape (batch,).
        """
        query = drug_emb.unsqueeze(1)  # (batch, 1, dim)
        attn_out, _ = self.attn(
            query, pheno_embs, pheno_embs, key_padding_mask=pheno_mask
        )  # (batch, 1, dim)
        score = self.score_proj(attn_out.squeeze(1)).squeeze(-1)  # (batch,)
        return score


class PhenoDrugModel(nn.Module):
    """End-to-end model: R-GCN layers + cross-attention scorer (flat layout).

    Parameter layout matches the training notebook so checkpoints load directly:
        node_emb.weight
        convs.0.*, convs.1.*, ...
        norms.0.*, norms.1.*, ...
        cross_attn.attn.*, cross_attn.score_proj.*

    Args:
        num_nodes: Total nodes in PrimeKG graph.
        num_relations: Number of relation types (including reverse edges).
        hidden_dim: Embedding dimensionality.
        num_bases: Basis decomposition rank for R-GCN weight sharing.
        num_layers: Number of R-GCN message-passing layers.
        num_heads: Cross-attention heads.
        dropout: Dropout rate (applied after each R-GCN layer and inside attention).
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_dim: int,
        num_bases: int = 10,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.cross_attn = DrugConditionedCrossAttention(hidden_dim, num_heads, dropout)
        self.dropout = dropout

    def encode(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """Full-graph R-GCN forward pass.

        Args:
            edge_index: (2, num_edges) COO edge indices.
            edge_type: (num_edges,) relation type per edge.

        Returns:
            Node embeddings of shape (num_nodes, hidden_dim).
        """
        x = self.node_emb.weight
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index, edge_type)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual
        return x

    def score(
        self,
        node_embs: torch.Tensor,
        drug_indices: torch.Tensor,
        pheno_indices: torch.Tensor,
        pheno_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score drugs against phenotype sets using cached node embeddings.

        Args:
            node_embs: (num_nodes, dim) from encode().
            drug_indices: (batch,) drug node indices.
            pheno_indices: (batch, max_phenos) padded phenotype node indices.
            pheno_mask: (batch, max_phenos) True where padded.

        Returns:
            Scalar scores of shape (batch,).
        """
        drug_emb = node_embs[drug_indices]
        pheno_embs = node_embs[pheno_indices]
        return self.cross_attn(drug_emb, pheno_embs, pheno_mask)
