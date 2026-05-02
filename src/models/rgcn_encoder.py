"""2-layer R-GCN encoder with basis decomposition over PrimeKG."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCNEncoder(nn.Module):
    """Full-graph R-GCN encoder that produces node embeddings.

    Args:
        num_nodes: Total nodes in PrimeKG graph.
        num_relations: Number of relation types (including reverse edges).
        hidden_dim: Embedding dimensionality.
        num_bases: Basis decomposition rank for R-GCN weight sharing.
        num_layers: Number of R-GCN message-passing layers.
        dropout: Dropout rate applied after each layer.
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_dim: int = 128,
        num_bases: int = 10,
        num_layers: int = 2,
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

        self.dropout = dropout

    def forward(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """Encode the full graph into node embeddings.

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
