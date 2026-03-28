"""Drug-conditioned cross-attention scorer.

Drug embedding = query, phenotype embeddings = keys/values.
Produces a scalar relevance score per (drug, phenotype-set) pair.
"""

from __future__ import annotations

import torch
import torch.nn as nn


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
    """End-to-end model: R-GCN encoder + cross-attention scorer.

    Args:
        encoder: RGCNEncoder instance.
        hidden_dim: Embedding dimension.
        num_heads: Cross-attention heads.
        dropout: Attention dropout.
    """

    def __init__(self, encoder: nn.Module, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = encoder
        self.cross_attn = DrugConditionedCrossAttention(hidden_dim, num_heads, dropout)

    def encode(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """Run the R-GCN encoder."""
        return self.encoder(edge_index, edge_type)

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
