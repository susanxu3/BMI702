"""Feature-level Graph-LLM fusion (Section 5.2.1, Method 2 of midterm report).

Two fusion approaches that replace raw node embeddings with fused
graph + text representations in the cross-attention scorer.

Text encoders (compared via late fusion first):
  PubMedBERT, BiomedBERT, BioLinkBERT, SPECTER2
All projected to 128 dims and L2-normalized before fusion.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata

logger = logging.getLogger(__name__)


class DegreeConditionedFusion(nn.Module):
    """Approach (a): Degree-conditioned weighted averaging.

    h_fused = alpha(v) * h_graph + (1 - alpha(v)) * h_LLM
    alpha(v) = sigmoid(w * log(deg(v) + 1) + b)

    Gate parameters (w, b) are frozen before ranking fine-tuning to prevent
    absorption of popularity bias. A semantic anchoring regularizer
    L_anc = ||h_fused - h_LLM||^2 prevents graph gradients from overwriting
    LLM signal for sparse nodes (following LLaDR, Xiang et al. 2025).

    Args:
        embed_dim: Shared dimensionality for graph and text embeddings (128).
    """

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))
        self.embed_dim = embed_dim

    def forward(
        self,
        h_graph: torch.Tensor,
        h_llm: torch.Tensor,
        log_degree: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse graph and text embeddings with degree-conditioned gating.

        Args:
            h_graph: (num_nodes, embed_dim) R-GCN node embeddings.
            h_llm: (num_nodes, embed_dim) cached text encoder embeddings.
            log_degree: (num_nodes,) log(deg(v) + 1) for each node.

        Returns:
            h_fused: (num_nodes, embed_dim) fused embeddings.
        """
        alpha = torch.sigmoid(self.w * log_degree + self.b).unsqueeze(-1)  # (N, 1)
        h_fused = alpha * h_graph + (1 - alpha) * h_llm
        return h_fused

    def anchoring_loss(self, h_fused: torch.Tensor, h_llm: torch.Tensor) -> torch.Tensor:
        """Semantic anchoring regularizer: ||h_fused - h_LLM||^2.

        Prevents graph gradients from overwriting LLM signal for sparse nodes.

        Args:
            h_fused: (num_nodes, embed_dim) fused embeddings.
            h_llm: (num_nodes, embed_dim) frozen text embeddings.

        Returns:
            Scalar anchoring loss.
        """
        return (h_fused - h_llm).pow(2).mean()

    def freeze_gate(self) -> None:
        """Freeze gate parameters (w, b) before ranking fine-tuning."""
        self.w.requires_grad_(False)
        self.b.requires_grad_(False)

    def unfreeze_gate(self) -> None:
        """Unfreeze gate parameters for gate pretraining phase."""
        self.w.requires_grad_(True)
        self.b.requires_grad_(True)


class AutoencoderFusion(nn.Module):
    """Approach (b): Autoencoder fusion (following LLM-DDA, Gu et al. 2025).

    Two-phase training:
      Phase 1: Unsupervised pretraining on ALL 129,375 PrimeKG nodes via
        reconstruction loss. Produces meaningful fused embeddings for cold-start
        drugs without requiring indication labels.
      Phase 2: Supervised fine-tuning with ranking loss, using the latent code
        in place of node embeddings in the cross-attention scorer.

    Args:
        input_dim: Concatenated input dimension (graph_dim + text_dim).
        hidden_dim: Intermediate encoder layer dimension.
        latent_dim: Bottleneck dimension (used as fused embedding).
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, h_graph: torch.Tensor, h_llm: torch.Tensor) -> torch.Tensor:
        """Encode concatenated embeddings to latent fused representation.

        Args:
            h_graph: (num_nodes, graph_dim) R-GCN node embeddings.
            h_llm: (num_nodes, text_dim) cached text encoder embeddings.

        Returns:
            Latent code of shape (num_nodes, latent_dim).
        """
        h_concat = torch.cat([h_graph, h_llm], dim=-1)
        return self.encoder(h_concat)

    def reconstruct(self, h_graph: torch.Tensor, h_llm: torch.Tensor) -> torch.Tensor:
        """Full autoencoder pass for reconstruction loss (Phase 1 pretraining).

        Args:
            h_graph: (num_nodes, graph_dim) R-GCN node embeddings.
            h_llm: (num_nodes, text_dim) cached text encoder embeddings.

        Returns:
            Reconstructed concatenated embeddings (num_nodes, input_dim).
        """
        h_concat = torch.cat([h_graph, h_llm], dim=-1)
        latent = self.encoder(h_concat)
        return self.decoder(latent)

    def reconstruction_loss(
        self, h_graph: torch.Tensor, h_llm: torch.Tensor
    ) -> torch.Tensor:
        """MSE reconstruction loss for unsupervised pretraining.

        Args:
            h_graph: (num_nodes, graph_dim) R-GCN node embeddings.
            h_llm: (num_nodes, text_dim) cached text encoder embeddings.

        Returns:
            Scalar reconstruction loss.
        """
        h_concat = torch.cat([h_graph, h_llm], dim=-1)
        reconstructed = self.reconstruct(h_graph, h_llm)
        return F.mse_loss(reconstructed, h_concat)


class LateFusion:
    """Late fusion baseline (Section 5.2.1, Method 1): no retraining.

    s_final(d) = beta * s_graph(d) + (1 - beta) * s_LLM(d)

    s_graph: scalar output from trained R-GCN cross-attention scorer.
    s_LLM: cosine similarity between mean-pooled phenotype text embedding
      and drug text embedding (pre-computed dot-product lookup).

    Args:
        beta: Mixing weight (tuned on validation MRR).
    """

    def __init__(self, beta: float = 0.5) -> None:
        self.beta = beta

    def fuse_scores(
        self,
        s_graph: torch.Tensor,
        s_llm: torch.Tensor,
    ) -> torch.Tensor:
        """Combine graph and LLM scores.

        Args:
            s_graph: (num_drugs,) graph model scores.
            s_llm: (num_drugs,) text embedding cosine similarities.

        Returns:
            Fused scores of shape (num_drugs,).
        """
        return self.beta * s_graph + (1 - self.beta) * s_llm

    @staticmethod
    def compute_llm_scores(
        pheno_embeds: torch.Tensor,
        drug_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute s_LLM as cosine similarity from cached embeddings.

        Args:
            pheno_embeds: (num_phenos, embed_dim) text embeddings of patient phenotypes.
            drug_embeds: (num_drugs, embed_dim) text embeddings of all drugs.

        Returns:
            Cosine similarities of shape (num_drugs,).
        """
        # Mean-pool phenotype embeddings
        pheno_mean = F.normalize(pheno_embeds.mean(dim=0, keepdim=True), dim=-1)
        drug_normed = F.normalize(drug_embeds, dim=-1)
        return (pheno_mean @ drug_normed.T).squeeze(0)

    @staticmethod
    def calibrate_beta(
        graph_scores: dict[int, np.ndarray],
        llm_scores: dict[int, np.ndarray],
        disease_to_true_drugs: dict[int, list[int]],
        drug_indices_arr: np.ndarray,
        beta_candidates: list[float] | None = None,
        normalize: str = "minmax",
        n_folds: int = 5,
        seed: int = 42,
    ) -> tuple[float, float, dict[float, float]]:
        """Calibrate beta via 5-fold CV over training diseases.

        Args:
            graph_scores: disease_idx -> (num_drugs,) graph model scores.
            llm_scores: disease_idx -> (num_drugs,) text embedding cosine sims.
            disease_to_true_drugs: disease_idx -> list of true drug indices.
            drug_indices_arr: Array of all drug node indices (fixed ordering).
            beta_candidates: List of beta values to try.
            normalize: Normalization method ('minmax' or 'rank').
            n_folds: Number of CV folds.
            seed: Random seed for fold shuffling.

        Returns:
            (best_beta, best_cv_mrr, beta_to_mrr) for wandb logging.
        """
        from src.evaluation.metrics import reciprocal_rank

        if beta_candidates is None:
            beta_candidates = [i / 10 for i in range(11)]

        # Get diseases present in both score dicts
        diseases = sorted(
            set(graph_scores.keys()) & set(llm_scores.keys())
            & set(disease_to_true_drugs.keys())
        )
        if not diseases:
            raise ValueError("No diseases found in all three input dicts")

        # Shuffle and split into folds
        rng = np.random.default_rng(seed)
        disease_arr = np.array(diseases)
        rng.shuffle(disease_arr)
        folds = np.array_split(disease_arr, n_folds)

        # For each beta, collect fold-level MRRs
        beta_to_fold_mrrs: dict[float, list[float]] = {b: [] for b in beta_candidates}

        for fold_idx, held_out in enumerate(folds):
            held_out_set = set(held_out.tolist())

            for beta in beta_candidates:
                mrr_sum = 0.0
                count = 0

                for d_idx in held_out_set:
                    s_graph = graph_scores[d_idx]
                    s_llm = llm_scores[d_idx]
                    true_drugs = disease_to_true_drugs[d_idx]

                    if not true_drugs:
                        continue

                    s_g_norm = normalize_scores(s_graph, method=normalize)
                    s_l_norm = normalize_scores(s_llm, method=normalize)
                    s_fused = beta * s_g_norm + (1 - beta) * s_l_norm

                    ranked = drug_indices_arr[np.argsort(-s_fused)].tolist()
                    mrr_sum += reciprocal_rank(ranked, true_drugs)
                    count += 1

                fold_mrr = mrr_sum / max(count, 1)
                beta_to_fold_mrrs[beta].append(fold_mrr)

            logger.info(f"  Fold {fold_idx + 1}/{n_folds} done ({len(held_out)} diseases)")

        # Average across folds
        beta_to_mrr = {
            b: float(np.mean(mrrs)) for b, mrrs in beta_to_fold_mrrs.items()
        }

        best_beta = max(beta_to_mrr, key=beta_to_mrr.get)  # type: ignore[arg-type]
        best_mrr = beta_to_mrr[best_beta]

        logger.info(f"Beta sweep results:")
        for b in sorted(beta_to_mrr.keys()):
            marker = " <-- best" if b == best_beta else ""
            logger.info(f"  beta={b:.1f}: MRR={beta_to_mrr[b]:.4f}{marker}")

        return best_beta, best_mrr, beta_to_mrr


# ── Score normalization ──────────────────────────────────────────────────
def normalize_scores(scores: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Normalize a per-disease score array to comparable scale.

    Applied independently to s_graph and s_LLM before mixing.

    Args:
        scores: (num_drugs,) raw scores for one disease.
        method: 'minmax' maps to [0,1]; 'rank' uses percentile rank.

    Returns:
        Normalized scores of same shape.
    """
    if method == "minmax":
        s_min = scores.min()
        s_max = scores.max()
        denom = s_max - s_min + 1e-8
        return (scores - s_min) / denom
    elif method == "rank":
        return rankdata(scores) / len(scores)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
