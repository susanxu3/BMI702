"""Feature-level Graph-LLM fusion training (Phase 3).

Two approaches that replace raw R-GCN node embeddings with fused graph + text
representations in the cross-attention scorer:

  (a) Degree-conditioned weighted averaging
      Phase 1 — analytical gate calibration (no gradients):
        - Pair-level quartile partition by KG-structural drug log_degree
        - Fixed-alpha sweep per quartile (one R-GCN encode, reused for all)
        - Closed-form linear regression on (median_log_deg, logit(alpha*))
        - Freeze gate immediately
      Phase 2 — supervised fine-tuning with frozen gate + mask-restricted
        anchoring loss (only nodes with cached text contribute).

  (b) Concat autoencoder fusion (LLM-DDA style)
      Phase 1 — unsupervised AE pretraining (reconstruction MSE on rows
        with cached text only).
      Phase 2 — supervised fine-tuning of R-GCN + cross-attention + AE encoder.

The orchestrator mirrors src/evaluation/late_fusion_eval.py: thin notebooks
import `run_degree_cond_experiment` / `run_autoencoder_experiment`.

All metrics flow through `late_fusion_eval.compute_test_metrics` so the metric
set is bit-identical to the late-fusion pipeline.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from time import perf_counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.evaluation.late_fusion_eval import (
    _extract_state_dict,
    _infer_arch_from_state_dict,
    _pick_num_heads,
    compute_test_metrics,
)
from src.evaluation.metrics import reciprocal_rank
from src.models.cross_attention_scorer import PhenoDrugModel
from src.models.fusion import (
    AutoencoderFusion,
    DegreeConditionedFusion,
    ResidualAutoencoderFusion,
)
from src.models.train import pad_pheno_batch
from src.utils.negative_sampling import DegreeWeightedSampler

logger = logging.getLogger(__name__)

OFF_LABEL_REL = "off-label use"


def _fmt_elapsed(start_time: float) -> str:
    """Human-readable elapsed time since `start_time`."""
    return f"{perf_counter() - start_time:.1f}s"


def _print_and_log(message: str) -> None:
    """Emit progress to both notebook stdout and the logger."""
    print(message, flush=True)
    logger.info(message)


def _cpu_clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Clone a module state_dict onto CPU for best-state checkpointing."""
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _build_runtime_graph(
    kg_df: pd.DataFrame,
    heldout_diseases: set[int],
    rel2id: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a masked runtime graph while preserving the original relation IDs."""
    from src.data.primekg_loader import DST_COL, REL_COL, SRC_COL, mask_test_diseases

    kg_masked = mask_test_diseases(kg_df, heldout_diseases)
    rel = np.array([rel2id[r] for r in kg_masked[REL_COL].values], dtype=np.int64)
    src = kg_masked[SRC_COL].to_numpy(dtype=np.int64)
    dst = kg_masked[DST_COL].to_numpy(dtype=np.int64)
    num_orig_rels = len(rel2id)

    edge_src = np.concatenate([src, dst])
    edge_dst = np.concatenate([dst, src])
    edge_type_np = np.concatenate([rel, rel + num_orig_rels])

    edge_index = torch.tensor(
        np.vstack([edge_src, edge_dst]), dtype=torch.long, device=device
    )
    edge_type = torch.tensor(edge_type_np, dtype=torch.long, device=device)
    return edge_index, edge_type


# ── Helpers ──────────────────────────────────────────────────────────────
def build_h_llm_full(
    drug_embed_path: str | Path,
    pheno_embed_path: str | Path,
    num_nodes: int,
    drug_indices_arr: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter cached drug + phenotype text embeddings into a full (N, D) tensor.

    For non-drug, non-phenotype nodes, h_llm rows are zero placeholders. The
    `MaskedFusionWrapper` ensures fused output equals h_graph for those rows,
    so the placeholder values are never propagated downstream.

    Args:
        drug_embed_path: Path to drug_embeddings.pt.
        pheno_embed_path: Path to phenotype_embeddings.pt.
        num_nodes: Total nodes in PrimeKG (e.g., 129,375).
        drug_indices_arr: Sorted array of all drug node indices (validated
            against the cached drug ordering).
        device: Compute device.

    Returns:
        h_llm_full: (num_nodes, embed_dim) tensor on `device`.
        has_text_mask: (num_nodes,) bool tensor, True where row was populated
            from a cached embedding.
    """
    t0 = perf_counter()
    logger.info(
        f"build_h_llm_full: loading drug embeddings from {drug_embed_path} "
        f"and phenotype embeddings from {pheno_embed_path}"
    )
    drug_data = torch.load(drug_embed_path, map_location="cpu", weights_only=False)
    pheno_data = torch.load(pheno_embed_path, map_location="cpu", weights_only=False)

    drug_embeds = drug_data["embeddings"]  # (num_drugs, dim)
    drug_indices = list(drug_data["node_indices"])
    pheno_embeds = pheno_data["embeddings"]  # (num_phenos, dim)
    pheno_indices = list(pheno_data["node_indices"])

    # Validate drug ordering matches drug_indices_arr (same check load_llm_scores does)
    expected = drug_indices_arr.tolist()
    if drug_indices != expected:
        mismatch = sum(1 for a, b in zip(drug_indices, expected) if a != b)
        raise ValueError(
            f"Drug embedding ordering does not match drug_indices_arr "
            f"(cached len={len(drug_indices)}, expected len={len(expected)}, "
            f"mismatched positions={mismatch}). Re-run cache_embeddings.py "
            f"or align orderings before fusing."
        )

    embed_dim = drug_embeds.shape[1]
    if pheno_embeds.shape[1] != embed_dim:
        raise ValueError(
            f"Drug embed dim {embed_dim} != pheno embed dim {pheno_embeds.shape[1]}"
        )

    h_llm_full = torch.zeros(num_nodes, embed_dim, dtype=torch.float32)
    has_text_mask = torch.zeros(num_nodes, dtype=torch.bool)

    drug_idx_t = torch.tensor(drug_indices, dtype=torch.long)
    h_llm_full[drug_idx_t] = drug_embeds.float()
    has_text_mask[drug_idx_t] = True

    pheno_idx_t = torch.tensor(pheno_indices, dtype=torch.long)
    h_llm_full[pheno_idx_t] = pheno_embeds.float()
    has_text_mask[pheno_idx_t] = True

    logger.info(
        f"build_h_llm_full: scattered embeddings in {_fmt_elapsed(t0)}; "
        f"shape={tuple(h_llm_full.shape)}, has_text_mask.sum()={int(has_text_mask.sum())}"
    )
    return h_llm_full.to(device), has_text_mask.to(device)


def compute_log_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute log(node_degree + 1) from a PyG edge_index.

    Counts each edge endpoint occurrence in the flattened (2*E,) tensor — i.e.
    treats edges as undirected for the degree count, which matches how the
    R-GCN actually uses the bidirectional `edge_index`.

    Args:
        edge_index: (2, num_edges) LongTensor.
        num_nodes: Total nodes.

    Returns:
        (num_nodes,) float tensor of log(deg + 1).
    """
    counts = torch.bincount(edge_index.flatten(), minlength=num_nodes).float()
    return torch.log1p(counts)


class MaskedFusionWrapper(nn.Module):
    """Wraps a fusion module so non-text rows pass through h_graph unchanged.

    For node v with `has_text_mask[v] == True`: output is the inner fusion's
    output. For non-text rows: output is `h_graph[v]`. This is semantically
    correct because the cached text embeddings cover only ~11.5K of 129,375
    PrimeKG nodes, and `PhenoDrugModel.score()` only reads drug + phenotype
    rows downstream — but the pass-through guarantees the fusion module never
    silently overwrites a non-text node's embedding with garbage derived from
    the zero-placeholder h_llm row.

    Args:
        fusion: A `DegreeConditionedFusion` or `AutoencoderFusion` instance.
        has_text_mask: (num_nodes,) bool tensor.
    """

    def __init__(self, fusion: nn.Module, has_text_mask: torch.Tensor) -> None:
        super().__init__()
        self.fusion = fusion
        self.register_buffer("mask", has_text_mask)

    def forward(
        self, h_graph: torch.Tensor, h_llm: torch.Tensor, *extra
    ) -> torch.Tensor:
        h_fused = self.fusion(h_graph, h_llm, *extra)
        return torch.where(self.mask.unsqueeze(-1), h_fused, h_graph)


def masked_anchoring(
    h_fused: torch.Tensor, h_llm: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Anchoring MSE on text-covered rows only.

    Computed inline here (not on `DegreeConditionedFusion`) so `src/models/fusion.py`
    stays untouched — it is also imported by the late-fusion pipeline. Keeping
    `fusion.py` unmodified avoids any risk of regressing that path.

    Args:
        h_fused: (N, D) fused embeddings.
        h_llm: (N, D) cached text embeddings (zero placeholder for non-text rows).
        mask: (N,) bool, True where row has cached text.

    Returns:
        Scalar MSE.
    """
    return ((h_fused[mask] - h_llm[mask]) ** 2).mean()


def split_train_val(
    train_diseases: set[int] | list[int],
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Runtime hold-out split of train diseases (no val split exists on disk).

    Args:
        train_diseases: Set / list of train disease node indices.
        val_frac: Fraction of train diseases held out for early stopping.
        seed: RNG seed.

    Returns:
        (train_subset, val_subset) sorted lists.
    """
    rng = np.random.default_rng(seed)
    arr = np.array(sorted(train_diseases))
    rng.shuffle(arr)
    n_val = max(1, int(len(arr) * val_frac))
    val_subset = sorted(arr[:n_val].tolist())
    train_subset = sorted(arr[n_val:].tolist())
    logger.info(
        f"split_train_val: train_subset={len(train_subset)}, "
        f"val_subset={len(val_subset)}, val_frac={val_frac}"
    )
    return train_subset, val_subset


def build_pheno_drug_model_from_checkpoint(
    checkpoint_path: str | Path,
    num_nodes: int,
    num_relations: int,
    config: dict,
    device: torch.device,
) -> tuple[PhenoDrugModel, dict[str, int]]:
    """Load PhenoDrugModel from a checkpoint, inferring arch from state_dict.

    Mirrors the loading logic in `late_fusion_eval.run_late_fusion_experiment`.

    Args:
        checkpoint_path: Path to .pt checkpoint.
        num_nodes: Total nodes in current graph.
        num_relations: Relations in current graph (must match checkpoint).
        config: Config dict (may contain hidden_dim/num_layers/num_bases/num_heads/dropout).
        device: Target device.

    Returns:
        (model, inferred_arch_dict).
    """
    checkpoint_obj = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = _extract_state_dict(checkpoint_obj)
    inferred = _infer_arch_from_state_dict(state_dict)

    arch_override = bool(config.get("arch_override", False))
    if arch_override:
        hidden_dim = int(config.get("hidden_dim", inferred["hidden_dim"]))
        num_layers = int(config.get("num_layers", inferred["num_layers"]))
        num_bases = int(config.get("num_bases", inferred.get("num_bases", 10)))
    else:
        for k in ["hidden_dim", "num_layers", "num_bases"]:
            if k in config and k in inferred and int(config[k]) != int(inferred[k]):
                logger.warning(
                    f"Config {k}={config[k]} mismatches checkpoint {k}={inferred[k]}; "
                    "using checkpoint value. Set arch_override=True to force config."
                )
        hidden_dim = int(inferred["hidden_dim"])
        num_layers = int(inferred["num_layers"])
        num_bases = int(inferred.get("num_bases", config.get("num_bases", 10)))

    num_heads = _pick_num_heads(hidden_dim, int(config.get("num_heads", 4)))
    dropout = float(config.get("dropout", 0.2))

    comp0 = state_dict.get("convs.0.comp")
    if comp0 is not None and hasattr(comp0, "shape") and len(comp0.shape) == 2:
        ckpt_num_rel = int(comp0.shape[0])
        if ckpt_num_rel != int(num_relations):
            raise ValueError(
                "Checkpoint num_relations mismatch: "
                f"checkpoint has {ckpt_num_rel}, current graph has {num_relations}."
            )

    model = PhenoDrugModel(
        num_nodes=num_nodes,
        num_relations=num_relations,
        hidden_dim=hidden_dim,
        num_bases=num_bases,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    inferred["num_heads"] = num_heads
    inferred["dropout"] = dropout
    return model, inferred


# ── Per-disease scoring with fused embeddings ────────────────────────────
def _score_diseases_with_fused_embeddings(
    model: PhenoDrugModel,
    h_fused: torch.Tensor,
    diseases: list[int],
    disease_to_phenotypes: dict[int, set[int]],
    drug_indices_arr: np.ndarray,
    device: torch.device,
    chunk_size: int = 512,
    progress_label: str | None = None,
) -> dict[int, np.ndarray]:
    """Run chunked drug scoring for each disease using already-fused embeddings.

    `h_fused` should be precomputed once outside this loop (encode → fuse).
    Mirrors `late_fusion_eval.load_graph_scores` chunking but takes fused
    embeddings directly instead of computing them inside.

    Args:
        model: PhenoDrugModel (only `.score` is invoked, never `.encode`).
        h_fused: (num_nodes, dim) fused node embeddings.
        diseases: Disease indices to score.
        disease_to_phenotypes: Disease -> phenotype indices.
        drug_indices_arr: Sorted (num_drugs,) array.
        device: Compute device.
        chunk_size: Drug chunk size.

    Returns:
        disease_idx -> (num_drugs,) numpy score array.
    """
    model.eval()
    all_drugs_t = torch.tensor(drug_indices_arr, dtype=torch.long, device=device)
    scores_dict: dict[int, np.ndarray] = {}

    t0 = perf_counter()
    if progress_label:
        logger.info(
            f"{progress_label}: scoring {len(diseases)} diseases across "
            f"{len(drug_indices_arr)} drugs (chunk_size={chunk_size})"
        )

    with torch.no_grad():
        progress_every = max(1, len(diseases) // 5) if diseases else 1
        for disease_num, d_idx in enumerate(diseases, start=1):
            phenos = list(disease_to_phenotypes.get(d_idx, []))
            if not phenos:
                continue
            chunk_scores = []
            for c_start in range(0, len(drug_indices_arr), chunk_size):
                c_end = min(c_start + chunk_size, len(drug_indices_arr))
                c_drugs = all_drugs_t[c_start:c_end]
                c_padded, c_mask = pad_pheno_batch(
                    [phenos] * (c_end - c_start), device
                )
                scores = model.score(h_fused, c_drugs, c_padded, c_mask)
                chunk_scores.append(scores.cpu().numpy())
            scores_dict[d_idx] = np.concatenate(chunk_scores)
            if progress_label and (
                disease_num == 1
                or disease_num == len(diseases)
                or disease_num % progress_every == 0
            ):
                logger.info(
                    f"{progress_label}: scored {disease_num}/{len(diseases)} diseases "
                    f"(elapsed {_fmt_elapsed(t0)})"
                )
    return scores_dict


def evaluate_fusion_model(
    model: PhenoDrugModel,
    masked_fusion: MaskedFusionWrapper,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    h_llm_full: torch.Tensor,
    log_degree: torch.Tensor | None,
    diseases: list[int],
    disease_to_phenotypes: dict[int, set[int]],
    disease_to_true_drugs: dict[int, list[int] | set[int]],
    drug_indices_arr: np.ndarray,
    device: torch.device,
    train_pairs: pd.DataFrame | None = None,
    bypass_fusion: bool = False,
    use_amp: bool = False,
    h_graph_override: torch.Tensor | None = None,
) -> tuple[dict[str, float], dict[int, np.ndarray]]:
    """Encode → fuse → chunked score → metrics. Returns (metrics, scores_dict).

    The scores_dict is returned alongside the metrics so callers can run a
    second metric pass with a different ground-truth dict (e.g., off-label-as-true)
    without recomputing scores.

    Args:
        bypass_fusion: If True, feed h_graph straight into scoring (skip
            masked_fusion). Used by the baseline-sanity check.
    """
    t0 = perf_counter()
    model.eval()
    logger.info(
        f"evaluate_fusion_model: start (diseases={len(diseases)}, "
        f"bypass_fusion={bypass_fusion}, use_amp={use_amp})"
    )
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp and device.type == "cuda",
        ):
            if h_graph_override is None:
                h_graph = model.encode(edge_index, edge_type)
            else:
                h_graph = h_graph_override
            if bypass_fusion:
                h_used = h_graph
            else:
                extras = (log_degree,) if isinstance(
                    masked_fusion.fusion, DegreeConditionedFusion
                ) else ()
                h_used = masked_fusion(h_graph, h_llm_full, *extras)

        scores_dict = _score_diseases_with_fused_embeddings(
            model, h_used, diseases, disease_to_phenotypes,
            drug_indices_arr, device,
            progress_label="evaluate_fusion_model",
        )

    metrics = compute_test_metrics(
        scores_dict=scores_dict,
        true_drugs_dict={d: list(v) for d, v in disease_to_true_drugs.items()},
        drug_indices_arr=drug_indices_arr,
        train_pairs=train_pairs,
    )
    logger.info(
        f"evaluate_fusion_model: finished in {_fmt_elapsed(t0)} "
        f"(MRR={metrics['MRR']:.4f}, R@10={metrics['R@10']:.4f})"
    )
    return metrics, scores_dict


def build_off_label_truth(
    kg_df: pd.DataFrame,
    test_diseases: set[int],
    truth_ind: dict[int, set[int]],
    drug_indices_arr: np.ndarray,
) -> dict[int, set[int]]:
    """Augment indication-only test GT with off-label-use edges from full KG.

    Mirrors Section 10 cell 62 of late_fusion_pipeline.ipynb. `kg_df` rows for
    `"off-label use"` are stored as `(x_index=drug, y_index=disease)`.
    """
    drug_set = set(drug_indices_arr.tolist())
    kg_off = kg_df[kg_df["relation"] == OFF_LABEL_REL][["x_index", "y_index"]].values
    off_by_disease: dict[int, set[int]] = defaultdict(set)
    for drug_idx, disease_idx in kg_off:
        d = int(disease_idx)
        drug = int(drug_idx)
        if d in test_diseases and drug in drug_set:
            off_by_disease[d].add(drug)
    truth_off = {
        d: truth_ind.get(d, set()) | off_by_disease.get(d, set())
        for d in test_diseases
    }
    added = sum(len(truth_off[d] - truth_ind.get(d, set())) for d in test_diseases)
    logger.info(
        f"Off-label GT: added {added} edges across "
        f"{sum(1 for d in test_diseases if off_by_disease.get(d))} diseases"
    )
    return truth_off


# ── Training step (shared between approaches) ────────────────────────────
def _training_step(
    model: PhenoDrugModel,
    masked_fusion: MaskedFusionWrapper,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    h_llm_full: torch.Tensor,
    has_text_mask: torch.Tensor,
    log_degree: torch.Tensor | None,
    batch_pairs: list[tuple[int, int]],
    disease_to_phenotypes: dict[int, set[int]],
    disease_to_true_drugs: dict[int, set[int]],
    sampler: DegreeWeightedSampler,
    margin: float,
    neg_ratio: int,
    anchoring_weight: float,
    use_anchoring: bool,
    device: torch.device,
    use_amp: bool,
    bypass_fusion: bool = False,
    h_graph_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """One forward pass producing the scalar loss (not yet backwarded)."""
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=use_amp and device.type == "cuda",
    ):
        if h_graph_override is None:
            h_graph = model.encode(edge_index, edge_type)
        else:
            h_graph = h_graph_override
        if bypass_fusion:
            h_fused = h_graph
        else:
            extras = (log_degree,) if isinstance(
                masked_fusion.fusion, DegreeConditionedFusion
            ) else ()
            h_fused = masked_fusion(h_graph, h_llm_full, *extras)

        pos_drug_ids = [p[1] for p in batch_pairs]
        pheno_lists = [list(disease_to_phenotypes[p[0]]) for p in batch_pairs]
        pos_drugs_t = torch.tensor(pos_drug_ids, dtype=torch.long, device=device)
        pheno_padded, pheno_mask = pad_pheno_batch(pheno_lists, device)
        pos_scores = model.score(h_fused, pos_drugs_t, pheno_padded, pheno_mask)

        # Sample negatives
        neg_drug_ids: list[int] = []
        for d_idx, pos_drug in batch_pairs:
            for n in sampler.sample(pos_drug, disease_to_true_drugs[d_idx], n=neg_ratio):
                neg_drug_ids.append(n)
        # If neg_ratio > 1, expand pheno lists / pos drugs for matching shapes
        if neg_ratio > 1:
            pheno_lists_neg = [
                list(disease_to_phenotypes[p[0]])
                for p in batch_pairs for _ in range(neg_ratio)
            ]
            neg_pheno_padded, neg_pheno_mask = pad_pheno_batch(pheno_lists_neg, device)
        else:
            neg_pheno_padded, neg_pheno_mask = pheno_padded, pheno_mask
        neg_drugs_t = torch.tensor(neg_drug_ids, dtype=torch.long, device=device)
        neg_scores = model.score(h_fused, neg_drugs_t, neg_pheno_padded, neg_pheno_mask)

        # Pair pos and neg scores. With neg_ratio negatives per pos, repeat pos to align.
        if neg_ratio > 1:
            pos_scores_rep = pos_scores.repeat_interleave(neg_ratio)
        else:
            pos_scores_rep = pos_scores
        target = torch.ones_like(pos_scores_rep)
        rank_loss = F.margin_ranking_loss(pos_scores_rep, neg_scores, target, margin=margin)

        loss = rank_loss
        if use_anchoring and anchoring_weight > 0:
            loss = loss + anchoring_weight * masked_anchoring(h_fused, h_llm_full, has_text_mask)
    return loss


# ── Approach (a): Degree-conditioned fusion ──────────────────────────────
def _phase1_analytical_calibration(
    model: PhenoDrugModel,
    fusion: DegreeConditionedFusion,
    masked_fusion: MaskedFusionWrapper,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    h_llm_full: torch.Tensor,
    log_degree: torch.Tensor,
    train_subset_diseases: list[int],
    train_disease_to_drugs: dict[int, set[int]],
    disease_to_phenotypes: dict[int, set[int]],
    drug_indices_arr: np.ndarray,
    alpha_grid: list[float],
    n_quartiles: int,
    device: torch.device,
) -> dict[str, object]:
    """Analytical gate calibration for DegreeConditionedFusion.

    Returns a dict with diagnostic info (quartile pairs, fitted w/b).
    """
    def phase1_print(message: str) -> None:
        print(message, flush=True)
        logger.info(message)

    t0 = perf_counter()
    phase1_print(
        f"Phase 1 calibration: start with {len(train_subset_diseases)} diseases, "
        f"alpha_grid={alpha_grid}, n_quartiles={n_quartiles}"
    )
    n_encode_calls = 0
    original_encode = model.encode

    def counting_encode(*args, **kwargs):
        nonlocal n_encode_calls
        n_encode_calls += 1
        return original_encode(*args, **kwargs)

    model.encode = counting_encode  # type: ignore[method-assign]
    try:
        # 1. KG structural log_degree already passed in as `log_degree`.
        log_deg_np = log_degree.detach().cpu().numpy()

        # 2. Build pair-level supervision list P (train subset only)
        train_subset_set = set(train_subset_diseases)
        P: list[tuple[int, int, float]] = []  # (disease, drug, x = log_degree[drug])
        for d in train_subset_diseases:
            for drug in train_disease_to_drugs.get(d, set()):
                P.append((int(d), int(drug), float(log_deg_np[int(drug)])))
        if not P:
            raise RuntimeError("Empty supervision pool P — no train pairs found")
        phase1_print(
            f"Phase 1: built supervision pool |P|={len(P)} "
            f"from {len(train_subset_diseases)} diseases (elapsed {_fmt_elapsed(t0)})"
        )

        # 3. Quartile-partition pairs by x
        xs = np.array([p[2] for p in P])
        quantiles = np.quantile(xs, np.linspace(0, 1, n_quartiles + 1))
        # Ensure boundaries are usable (np.digitize uses bins[1:])
        # bins for digitize: interior boundaries
        interior_bounds = quantiles[1:-1]
        quartile_assignment = np.digitize(xs, interior_bounds, right=False)  # 0..n_quartiles-1

        Q: list[list[tuple[int, int, float]]] = [[] for _ in range(n_quartiles)]
        for pair, q_idx in zip(P, quartile_assignment.tolist()):
            Q[q_idx].append(pair)
        m_q_list = [float(np.median([p[2] for p in q])) if q else float(np.nan) for q in Q]
        phase1_print(
            f"Phase 1: quartile sizes = {[len(q) for q in Q]}, m_q = {m_q_list} "
            f"(elapsed {_fmt_elapsed(t0)})"
        )

        # 4. Single encode pass — h_graph cached for the entire sweep.
        # Use the counted wrapper here so the verification step downstream
        # accurately reports n_encode_calls == 1.
        with torch.no_grad():
            h_graph = model.encode(edge_index, edge_type).detach()
        phase1_print(
            f"Phase 1: cached single h_graph encode (elapsed {_fmt_elapsed(t0)})"
        )

        optimal_alphas: list[float] = []
        for q_idx, q_pairs in enumerate(Q):
            if not q_pairs:
                phase1_print(
                    f"Phase 1: quartile {q_idx + 1}/{n_quartiles} is empty, defaulting alpha*=0.5"
                )
                optimal_alphas.append(0.5)
                continue
            D_q = sorted({p[0] for p in q_pairs})
            phase1_print(
                f"Phase 1: quartile {q_idx + 1}/{n_quartiles} start "
                f"(|pairs|={len(q_pairs)}, |diseases|={len(D_q)}, "
                f"m_q={m_q_list[q_idx]:.4f}, elapsed {_fmt_elapsed(t0)})"
            )
            mrr_per_alpha: dict[float, float] = {}
            for alpha in alpha_grid:
                alpha_t0 = perf_counter()
                phase1_print(
                    f"Phase 1: quartile {q_idx + 1}/{n_quartiles}, alpha={alpha:.2f} start"
                )
                with torch.no_grad():
                    h_fused_inner = alpha * h_graph + (1 - alpha) * h_llm_full
                    h_fused = torch.where(
                        masked_fusion.mask.unsqueeze(-1), h_fused_inner, h_graph
                    )
                    scores_dict = _score_diseases_with_fused_embeddings(
                        model, h_fused, D_q, disease_to_phenotypes,
                        drug_indices_arr, device,
                        progress_label=(
                            f"Phase 1 q{q_idx + 1}/{n_quartiles} alpha={alpha:.2f}"
                        ),
                    )
                # Pair-level MRR
                rr_sum = 0.0
                count = 0
                for d, drug, _ in q_pairs:
                    if d not in scores_dict:
                        continue
                    s = scores_dict[d]
                    ranked = drug_indices_arr[np.argsort(-s)].tolist()
                    rr_sum += reciprocal_rank(ranked, [drug])
                    count += 1
                mrr_per_alpha[alpha] = rr_sum / max(count, 1)
                phase1_print(
                    f"Phase 1: quartile {q_idx + 1}/{n_quartiles}, alpha={alpha:.2f} "
                    f"done in {_fmt_elapsed(alpha_t0)} "
                    f"(pair_mrr={mrr_per_alpha[alpha]:.4f})"
                )
            best_alpha = max(mrr_per_alpha, key=mrr_per_alpha.get)  # type: ignore[arg-type]
            optimal_alphas.append(float(best_alpha))
            phase1_print(
                f"Phase 1: q={q_idx} m_q={m_q_list[q_idx]:.4f} "
                f"alpha*={best_alpha:.2f} MRR={mrr_per_alpha[best_alpha]:.4f} "
                f"(elapsed {_fmt_elapsed(t0)})"
            )

        # 5. Closed-form linear regression on (m_q, logit(alpha*_q))
        eps = 1e-3
        clipped = [min(max(a, eps), 1 - eps) for a in optimal_alphas]
        y_q = [float(np.log(a / (1 - a))) for a in clipped]
        # Drop any NaN m_q (empty quartiles)
        valid = [(m, y) for m, y in zip(m_q_list, y_q) if not np.isnan(m)]
        if len(valid) < 2:
            raise RuntimeError("Phase 1: < 2 valid quartiles — cannot fit (w, b)")
        xs_fit = np.array([v[0] for v in valid])
        ys_fit = np.array([v[1] for v in valid])
        w_fit, b_fit = np.polyfit(xs_fit, ys_fit, 1)
        w_fit = float(w_fit)
        b_fit = float(b_fit)

        # 5b. Tiered monotonicity check (alpha non-decreasing across quartiles)
        n_violations = sum(
            1 for i in range(n_quartiles - 1)
            if optimal_alphas[i] > optimal_alphas[i + 1] + 0.05
        )
        diag_pairs = list(zip(m_q_list, optimal_alphas))
        if n_violations >= 1 or w_fit <= 0:
            logger.warning(
                "Gate calibration produced non-monotonic / borderline result: "
                f"n_violations={n_violations}, pairs={diag_pairs}, "
                f"(w, b)=({w_fit:.4f}, {b_fit:.4f}). "
                "Proceeding to Phase 2 but flagging for review."
            )

        # 6. Assign and freeze
        with torch.no_grad():
            fusion.w.data.fill_(w_fit)
            fusion.b.data.fill_(b_fit)
        fusion.freeze_gate()

        diag = {
            "m_q": m_q_list,
            "optimal_alphas": optimal_alphas,
            "w_fit": w_fit,
            "b_fit": b_fit,
            "n_encode_calls": n_encode_calls,
            "n_violations": n_violations,
        }
        phase1_print(
            f"Phase 1 calibration: finished in {_fmt_elapsed(t0)} "
            f"(w={w_fit:.4f}, b={b_fit:.4f}, encode_calls={n_encode_calls})"
        )
        return diag
    finally:
        model.encode = original_encode  # type: ignore[method-assign]


def _supervised_finetune(
    model: PhenoDrugModel,
    masked_fusion: MaskedFusionWrapper,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    h_llm_full: torch.Tensor,
    has_text_mask: torch.Tensor,
    log_degree: torch.Tensor | None,
    train_subset_diseases: list[int],
    val_subset_diseases: list[int],
    train_disease_to_drugs: dict[int, set[int]],
    disease_to_phenotypes: dict[int, set[int]],
    drug_indices_arr: np.ndarray,
    train_pairs_subset: pd.DataFrame,
    config: dict,
    use_anchoring: bool,
    device: torch.device,
    wandb_run=None,
    extra_trainable_params: list[nn.Parameter] | None = None,
) -> dict[str, float]:
    """Phase-2 supervised fine-tuning with ranking loss + (optional) anchoring."""
    import wandb
    t0 = perf_counter()

    sampler = DegreeWeightedSampler(
        train_pairs_subset, sorted(drug_indices_arr.tolist()),
        seed=int(config.get("seed", 42)),
    )

    train_scope = str(config.get("phase2_train_scope", "full_model"))
    bypass_fusion = bool(config.get("phase2_bypass_fusion", False))
    if train_scope not in {"full_model", "scorer_only"}:
        raise ValueError(
            f"Unsupported phase2_train_scope={train_scope!r}; "
            "expected 'full_model' or 'scorer_only'."
        )

    if train_scope == "scorer_only":
        for p in model.node_emb.parameters():
            p.requires_grad_(False)
        for module in list(model.convs) + list(model.norms):
            for p in module.parameters():
                p.requires_grad_(False)
        for p in model.cross_attn.parameters():
            p.requires_grad_(True)
        trainable: list[nn.Parameter] = [
            p for p in model.cross_attn.parameters() if p.requires_grad
        ]
    else:
        for p in model.parameters():
            p.requires_grad_(True)
        trainable = [p for p in model.parameters() if p.requires_grad]

    # Trainable params: selected PhenoDrugModel params plus any AE encoder
    # params passed in via extra_trainable_params.
    if extra_trainable_params:
        trainable.extend(p for p in extra_trainable_params if p.requires_grad)

    optimizer = torch.optim.Adam(
        trainable,
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )

    finetune_epochs = int(config.get("finetune_epochs", 100))
    patience = int(config.get("patience", 15))
    margin = float(config.get("margin", 1.0))
    neg_ratio = int(config.get("neg_ratio", 1))
    anchoring_weight = float(config.get("anchoring_loss_weight", 0.1))
    accum_steps = int(config.get("accum_steps", 4))
    grad_clip = float(config.get("grad_clip", 1.0))
    batch_size = int(config.get("batch_size", 512))
    use_amp = bool(config.get("use_amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    effective_use_anchoring = use_anchoring and not bypass_fusion
    _print_and_log(
        f"Phase 2 setup: train_scope={train_scope}, bypass_fusion={bypass_fusion}, "
        f"trainable_tensors={len(trainable)}, use_amp={use_amp}, "
        f"use_anchoring={effective_use_anchoring}, "
        f"epochs={finetune_epochs}, batch_size={batch_size}, lr={float(config.get('lr', 1e-3)):.1e}"
    )

    # Build pair list once
    pairs = [
        (int(r["disease_id"]), int(r["drug_id"]))
        for _, r in train_pairs_subset.iterrows()
    ]
    total_batches = max(1, (len(pairs) + batch_size - 1) // batch_size)
    rng = np.random.default_rng(int(config.get("seed", 42)))

    # Sanity log on the first batch — ensures masked vs unmasked anchoring differ
    masking_check_done = False

    best_val_mrr = -1.0
    best_state: dict[str, dict[str, torch.Tensor]] | None = None
    best_epoch = 0
    no_improve = 0
    epochs_ran = 0
    restored_best_state = False

    h_graph_frozen: torch.Tensor | None = None
    if train_scope == "scorer_only":
        prev_training = model.training
        model.eval()
        with torch.no_grad():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp and device.type == "cuda",
            ):
                h_graph_frozen = model.encode(edge_index, edge_type).detach()
        model.train(prev_training)
        _print_and_log(
            "Phase 2 setup: cached frozen h_graph once for scorer-only fine-tuning"
        )

    for epoch in range(finetune_epochs):
        epoch_t0 = perf_counter()
        _print_and_log(
            f"Phase 2: epoch {epoch + 1}/{finetune_epochs} start "
            f"(best_val_mrr={best_val_mrr:.4f}, no_improve={no_improve})"
        )
        model.train()
        rng.shuffle(pairs)
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        accum_count = 0
        epochs_ran = epoch + 1

        for b_start in range(0, len(pairs), batch_size):
            batch = pairs[b_start: b_start + batch_size]
            batch_num = n_batches + 1
            batch_t0 = perf_counter()
            loss = _training_step(
                model=model,
                masked_fusion=masked_fusion,
                edge_index=edge_index,
                edge_type=edge_type,
                h_llm_full=h_llm_full,
                has_text_mask=has_text_mask,
                log_degree=log_degree,
                batch_pairs=batch,
                disease_to_phenotypes=disease_to_phenotypes,
                disease_to_true_drugs=train_disease_to_drugs,
                sampler=sampler,
                margin=margin,
                neg_ratio=neg_ratio,
                anchoring_weight=anchoring_weight,
                use_anchoring=effective_use_anchoring,
                device=device,
                use_amp=use_amp,
                bypass_fusion=bypass_fusion,
                h_graph_override=h_graph_frozen,
            )

            scaler.scale(loss / accum_steps).backward()
            accum_count += 1
            if accum_count == accum_steps:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                accum_count = 0

            epoch_loss += float(loss.detach().cpu().item())
            n_batches += 1
            logger.info(
                f"Phase 2: epoch {epoch + 1}/{finetune_epochs}, "
                f"batch {batch_num}/{total_batches} done in {_fmt_elapsed(batch_t0)} "
                f"(loss={float(loss.detach().cpu().item()):.4f})"
            )

            if not masking_check_done and effective_use_anchoring and epoch == 0 and n_batches == 1:
                masking_check_done = True
                with torch.no_grad():
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=use_amp and device.type == "cuda",
                    ):
                        if h_graph_frozen is None:
                            h_graph_dbg = model.encode(edge_index, edge_type)
                        else:
                            h_graph_dbg = h_graph_frozen
                        if bypass_fusion:
                            h_fused_dbg = h_graph_dbg
                        else:
                            extras_dbg = (log_degree,) if log_degree is not None else ()
                            h_fused_dbg = masked_fusion(
                                h_graph_dbg, h_llm_full, *extras_dbg
                            )
                    masked_val = masked_anchoring(h_fused_dbg, h_llm_full, has_text_mask)
                    unmasked_val = ((h_fused_dbg - h_llm_full) ** 2).mean()
                logger.info(
                    f"Anchoring sanity: n_rows_in_anchoring={int(has_text_mask.sum())}, "
                    f"masked={float(masked_val):.6f}, unmasked={float(unmasked_val):.6f}, "
                    f"diff={abs(float(masked_val) - float(unmasked_val)):.6f}"
                )

        # Flush any remaining accumulated gradient
        if accum_count > 0:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Validation — restrict ground truth to the val subset's positives
        val_truth = {
            d: train_disease_to_drugs[d] for d in val_subset_diseases
            if d in train_disease_to_drugs
        }
        val_metrics, _ = evaluate_fusion_model(
            model=model,
            masked_fusion=masked_fusion,
            edge_index=edge_index,
            edge_type=edge_type,
            h_llm_full=h_llm_full,
            log_degree=log_degree,
            diseases=val_subset_diseases,
            disease_to_phenotypes=disease_to_phenotypes,
            disease_to_true_drugs=val_truth,
            drug_indices_arr=drug_indices_arr,
            device=device,
            use_amp=use_amp,
            bypass_fusion=bypass_fusion,
            h_graph_override=h_graph_frozen,
        )
        train_loss_avg = epoch_loss / max(n_batches, 1)
        logger.info(
            f"Epoch {epoch+1}/{finetune_epochs}: "
            f"train_loss={train_loss_avg:.4f} val_MRR={val_metrics['MRR']:.4f} "
            f"(epoch_elapsed={_fmt_elapsed(epoch_t0)}, total_elapsed={_fmt_elapsed(t0)})"
        )
        _print_and_log(
            f"Phase 2: epoch {epoch + 1}/{finetune_epochs} done "
            f"(train_loss={train_loss_avg:.4f}, val_MRR={val_metrics['MRR']:.4f}, "
            f"val_R@10={val_metrics['R@10']:.4f}, elapsed={_fmt_elapsed(epoch_t0)})"
        )
        if wandb_run is not None:
            wandb.log({
                "train/loss": train_loss_avg,
                "val/MRR": val_metrics["MRR"],
                "val/R@10": val_metrics["R@10"],
                "epoch": epoch,
            })

        if val_metrics["MRR"] > best_val_mrr:
            best_val_mrr = val_metrics["MRR"]
            best_epoch = epoch + 1
            best_state = {
                "model": _cpu_clone_state_dict(model),
                "fusion": _cpu_clone_state_dict(masked_fusion.fusion),
            }
            no_improve = 0
            _print_and_log(
                f"Phase 2: new best checkpoint at epoch {best_epoch} "
                f"(val_MRR={best_val_mrr:.4f})"
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                _print_and_log(
                    f"Phase 2: early stopping at epoch {epoch + 1} "
                    f"(patience={patience}, best_epoch={best_epoch}, "
                    f"best_val_mrr={best_val_mrr:.4f})"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state["model"], strict=True)
        masked_fusion.fusion.load_state_dict(best_state["fusion"], strict=True)
        restored_best_state = True
    _print_and_log(
        f"Phase 2 supervised finetune: finished in {_fmt_elapsed(t0)} "
        f"(best_epoch={best_epoch}, epochs_ran={epochs_ran}, "
        f"best_val_mrr={best_val_mrr:.4f}, restored_best_state={restored_best_state}, "
        f"use_amp={use_amp})"
    )
    return {
        "best_val_mrr": best_val_mrr,
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "restored_best_state": restored_best_state,
        "train_scope": train_scope,
        "bypass_fusion": bypass_fusion,
    }


def _setup_data(config: dict, device: torch.device) -> dict:
    """Shared data loading: PrimeKG + split + masked graph + supervision maps."""
    from src.data.disease_split import load_split
    from src.data.primekg_loader import (
        build_pyg_graph,
        build_supervision_maps,
        load_primekg,
        mask_test_diseases,
    )

    t0 = perf_counter()
    logger.info("setup_data: loading PrimeKG and split files")
    nodes_df, edges_df, kg_df = load_primekg(config["data_dir"])
    train_diseases, test_diseases, train_pairs, test_pairs = load_split(config["split_dir"])
    logger.info(
        f"setup_data: loaded nodes={len(nodes_df)}, kg_rows={len(kg_df)}, "
        f"train_diseases={len(train_diseases)}, test_diseases={len(test_diseases)} "
        f"in {_fmt_elapsed(t0)}"
    )
    kg_train = mask_test_diseases(kg_df, test_diseases)
    logger.info(
        f"setup_data: masked test diseases, remaining kg rows={len(kg_train)} "
        f"(elapsed {_fmt_elapsed(t0)})"
    )
    edge_index, edge_type, num_relations, rel2id = build_pyg_graph(kg_train, device)
    logger.info(
        f"setup_data: built edge_index shape={tuple(edge_index.shape)}, "
        f"num_relations={num_relations} (elapsed {_fmt_elapsed(t0)})"
    )

    supervision = build_supervision_maps(
        kg_df, nodes_df, train_diseases, test_diseases, train_pairs, test_pairs
    )
    drug_indices_arr = np.array(sorted(supervision["drug_indices"]))
    logger.info(
        f"setup_data: built supervision maps with {len(drug_indices_arr)} drugs "
        f"(elapsed {_fmt_elapsed(t0)})"
    )

    return {
        "nodes_df": nodes_df,
        "kg_df": kg_df,
        "train_diseases": train_diseases,
        "test_diseases": test_diseases,
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "num_relations": num_relations,
        "rel2id": rel2id,
        "supervision": supervision,
        "drug_indices_arr": drug_indices_arr,
        "num_nodes": len(nodes_df),
    }


def _log_test_metrics(wandb_run, prefix: str, metrics: dict[str, float]) -> None:
    import wandb
    for key, val in metrics.items():
        wandb.summary[f"{prefix}/{key}"] = val


def _per_disease_csv(
    scores_ind: dict[int, np.ndarray],
    scores_off: dict[int, np.ndarray],
    truth_ind: dict[int, set[int]],
    truth_off: dict[int, set[int]],
    drug_indices_arr: np.ndarray,
    disease_to_phenotypes: dict[int, set[int]],
    out_path: Path,
) -> pd.DataFrame:
    """Write per-disease CSV with both indication-only and off-label-aug metrics."""
    from src.evaluation.metrics import recall_at_k

    rows = []
    for d in sorted(scores_ind.keys()):
        s = scores_ind[d]
        ranked = drug_indices_arr[np.argsort(-s)].tolist()
        true_ind_drugs = list(truth_ind.get(d, set()))
        true_off_drugs = list(truth_off.get(d, set()))
        if not true_ind_drugs:
            continue
        rows.append({
            "disease_idx": d,
            "n_phenotypes": len(disease_to_phenotypes.get(d, set())),
            "n_true_drugs_ind": len(true_ind_drugs),
            "n_true_drugs_off": len(true_off_drugs),
            "MRR": reciprocal_rank(ranked, true_ind_drugs),
            "R@10": recall_at_k(ranked, true_ind_drugs, 10),
            "MRR_off": reciprocal_rank(ranked, true_off_drugs) if true_off_drugs else 0.0,
            "R@10_off": recall_at_k(ranked, true_off_drugs, 10) if true_off_drugs else 0.0,
        })
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved per-disease CSV to {out_path}")
    return df


def run_degree_cond_experiment(config: dict) -> dict:
    """Top-level entry: degree-conditioned fusion (Approach a).

    Expected config keys (in addition to those documented in
    `late_fusion_eval.run_late_fusion_experiment`):
        encoder_name, desc_tier, projection, embed_dir, checkpoint_path,
        embed_dim (256), alpha_sweep_grid, n_quartiles, val_frac,
        finetune_epochs, patience, anchoring_loss_weight, lr, weight_decay,
        margin, neg_ratio, accum_steps, grad_clip, batch_size,
        phase2_train_scope, phase2_bypass_fusion, seed, device, results_dir.
    """
    import wandb
    t0 = perf_counter()

    encoder_name = config["encoder_name"]
    desc_tier = config["desc_tier"]
    projection = config["projection"]
    device = torch.device(config.get("device", "cuda"))

    run_name = f"feature-fusion-degcond-{encoder_name}-{desc_tier}-{projection}"
    wandb.init(project="rare-disease-repurposing", name=run_name)
    wandb.config.update(config)
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    logger.info(
        f"run_degree_cond_experiment: start encoder={encoder_name}, "
        f"tier={desc_tier}, projection={projection}, device={device}"
    )

    # Data
    data = _setup_data(config, device)
    eval_edge_index, eval_edge_type = data["edge_index"], data["edge_type"]
    drug_indices_arr = data["drug_indices_arr"]
    num_nodes = data["num_nodes"]
    sup = data["supervision"]
    train_disease_to_drugs = sup["train_disease_to_drugs"]
    test_disease_to_drugs = sup["test_disease_to_drugs"]
    disease_to_phenotypes = sup["disease_to_phenotypes"]

    # Val split (BEFORE Phase 1)
    train_subset, val_subset = split_train_val(
        data["train_diseases"],
        val_frac=float(config.get("val_frac", 0.1)),
        seed=int(config.get("seed", 42)),
    )
    assert set(train_subset).isdisjoint(set(val_subset))
    train_pairs_subset = data["train_pairs"][
        data["train_pairs"]["disease_id"].isin(train_subset)
    ].reset_index(drop=True)
    print(f"Train subset: {len(train_subset)} diseases, {len(train_pairs_subset)} pairs")
    print(f"Val subset: {len(val_subset)} diseases")
    train_graph_heldout = set(data["test_diseases"]) | set(val_subset)
    train_edge_index, train_edge_type = _build_runtime_graph(
        data["kg_df"], train_graph_heldout, data["rel2id"], device,
    )
    print(
        "Rebuilt train_graph with held-out diseases masked: "
        f"{len(train_graph_heldout)} diseases"
    )
    logger.info(
        "run_degree_cond_experiment: rebuilt train_graph after runtime val split "
        f"(heldout_diseases={len(train_graph_heldout)}, "
        f"edge_index_shape={tuple(train_edge_index.shape)})"
    )

    # Model
    model, inferred = build_pheno_drug_model_from_checkpoint(
        config["checkpoint_path"], num_nodes, data["num_relations"], config, device,
    )
    assert model.node_emb.weight.shape[1] == int(config.get("embed_dim", 256)), (
        f"Checkpoint hidden_dim {model.node_emb.weight.shape[1]} "
        f"!= config embed_dim {config.get('embed_dim')}"
    )
    print("RGCN model loaded from checkpoint")

    # h_llm_full + has_text_mask
    print("Loading LLM-derived embeddings for h_llm_full construction...")
    embed_base = Path(config["embed_dir"]) / encoder_name / desc_tier / projection
    drug_path = embed_base / "drug_embeddings.pt"
    pheno_path = embed_base / "phenotype_embeddings.pt"
    if not drug_path.exists() or not pheno_path.exists():
        raise FileNotFoundError(f"Missing embeddings under {embed_base}")
    h_llm_full, has_text_mask = build_h_llm_full(
        drug_path, pheno_path, num_nodes, drug_indices_arr, device,
    )
    assert h_llm_full.shape == (num_nodes, model.node_emb.weight.shape[1]), (
        f"h_llm_full shape {h_llm_full.shape} mismatches model dim "
        f"{model.node_emb.weight.shape[1]}. Cached embeddings must match the "
        "R-GCN hidden_dim — check that embed_dir points at data/embeddings_256."
    )

    print("computing log_degree for degree-conditioned fusion...")
    train_log_degree = compute_log_degree(train_edge_index, num_nodes).to(device)
    eval_log_degree = compute_log_degree(eval_edge_index, num_nodes).to(device)
    logger.info(
        f"run_degree_cond_experiment: computed train/eval log_degree in {_fmt_elapsed(t0)}"
    )
    print("== Data and model setup complete ==")

    # Build fusion + wrapper
    embed_dim = int(config.get("embed_dim", 256))
    fusion = DegreeConditionedFusion(embed_dim=embed_dim).to(device)
    masked_fusion = MaskedFusionWrapper(fusion, has_text_mask).to(device)
    use_amp = bool(config.get("use_amp", True)) and device.type == "cuda"

    # ── Phase 1: Analytical gate calibration ──
    if bool(config.get("skip_phase1", False)):
        phase1_w = float(config.get("phase1_override_w", 0.0))
        phase1_b = float(config.get("phase1_override_b", 0.0))
        print(
            "Phase 1 skipped for smoke test; "
            f"using fixed gate w={phase1_w:.4f}, b={phase1_b:.4f}",
            flush=True,
        )
        with torch.no_grad():
            fusion.w.data.fill_(phase1_w)
            fusion.b.data.fill_(phase1_b)
        fusion.freeze_gate()
        diag = {
            "m_q": [],
            "optimal_alphas": [],
            "w_fit": phase1_w,
            "b_fit": phase1_b,
            "n_encode_calls": 0,
            "n_violations": 0,
            "skipped": True,
        }
        logger.warning(
            "Phase 1 skipped via config; using fixed gate "
            f"(w={phase1_w:.4f}, b={phase1_b:.4f}) for Phase 2 smoke testing."
        )
    else:
        print(
            "Phase 1: Sweeping alpha to calibrate gate parameters "
            "(w, b) for degree-conditioned fusion...",
            flush=True,
        )
        encoder_param_snapshot = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

        diag = _phase1_analytical_calibration(
            model=model, fusion=fusion, masked_fusion=masked_fusion,
            edge_index=train_edge_index, edge_type=train_edge_type,
            h_llm_full=h_llm_full, log_degree=train_log_degree,
            train_subset_diseases=train_subset,
            train_disease_to_drugs=train_disease_to_drugs,
            disease_to_phenotypes=disease_to_phenotypes,
            drug_indices_arr=drug_indices_arr,
            alpha_grid=list(config.get("alpha_sweep_grid", [i / 10 for i in range(11)])),
            n_quartiles=int(config.get("n_quartiles", 4)),
            device=device,
        )
        # Verify encoder untouched
        for k, v in model.state_dict().items():
            assert torch.equal(v, encoder_param_snapshot[k]), (
                f"Phase 1 modified encoder param {k}"
            )
        # Verify single encode pass (verification step 16)
        assert diag["n_encode_calls"] == 1, (
            f"Phase 1 ran model.encode {diag['n_encode_calls']} times (expected 1) — "
            "h_graph cache is being bypassed inside the alpha sweep."
        )

    # Verify gate frozen
    assert not fusion.w.requires_grad and not fusion.b.requires_grad
    wandb.summary["phase1/w_fit"] = diag["w_fit"]
    wandb.summary["phase1/b_fit"] = diag["b_fit"]
    wandb.summary["phase1/n_encode_calls"] = diag["n_encode_calls"]
    wandb.summary["phase1/n_violations"] = diag["n_violations"]
    for i, (m, a) in enumerate(zip(diag["m_q"], diag["optimal_alphas"])):
        wandb.summary[f"phase1/q{i}_m"] = m
        wandb.summary[f"phase1/q{i}_alpha_star"] = a
    logger.info(
        f"Phase 1 complete: w={diag['w_fit']:.4f} b={diag['b_fit']:.4f} "
        f"n_encode_calls={diag['n_encode_calls']} (expected 1)"
    )

    # ── Phase 2: Supervised fine-tuning (gate frozen) ──
    finetune_diag = _supervised_finetune(
        model=model, masked_fusion=masked_fusion,
        edge_index=train_edge_index, edge_type=train_edge_type,
        h_llm_full=h_llm_full, has_text_mask=has_text_mask,
        log_degree=train_log_degree,
        train_subset_diseases=train_subset, val_subset_diseases=val_subset,
        train_disease_to_drugs=train_disease_to_drugs,
        disease_to_phenotypes=disease_to_phenotypes,
        drug_indices_arr=drug_indices_arr,
        train_pairs_subset=train_pairs_subset,
        config=config, use_anchoring=True, device=device,
        wandb_run=wandb.run,
    )
    wandb.summary["best_val_mrr"] = finetune_diag["best_val_mrr"]
    wandb.summary["best_epoch"] = finetune_diag["best_epoch"]
    wandb.summary["epochs_ran"] = finetune_diag["epochs_ran"]
    wandb.summary["restored_best_state"] = int(finetune_diag["restored_best_state"])
    wandb.summary["phase2/train_scope"] = finetune_diag["train_scope"]
    wandb.summary["phase2/bypass_fusion"] = int(finetune_diag["bypass_fusion"])
    msg = (
        f"run_degree_cond_experiment: Phase 2 complete "
        f"(best_epoch={finetune_diag['best_epoch']}, "
        f"epochs_ran={finetune_diag['epochs_ran']}, "
        f"best_val_mrr={finetune_diag['best_val_mrr']:.4f}, "
        f"restored_best_state={finetune_diag['restored_best_state']}, "
        f"elapsed {_fmt_elapsed(t0)})"
    )
    print(msg, flush=True)
    logger.info(msg)

    # ── Test eval with both GT settings ──
    bypass_fusion = bool(config.get("phase2_bypass_fusion", False))
    test_metrics_ind, scores_test = evaluate_fusion_model(
        model=model, masked_fusion=masked_fusion,
        edge_index=eval_edge_index, edge_type=eval_edge_type,
        h_llm_full=h_llm_full, log_degree=eval_log_degree,
        diseases=sorted(data["test_diseases"]),
        disease_to_phenotypes=disease_to_phenotypes,
        disease_to_true_drugs={d: list(v) for d, v in test_disease_to_drugs.items()},
        drug_indices_arr=drug_indices_arr,
        device=device,
        train_pairs=data["train_pairs"],
        use_amp=use_amp,
        bypass_fusion=bypass_fusion,
    )
    truth_off = build_off_label_truth(
        data["kg_df"], data["test_diseases"],
        {d: set(v) for d, v in test_disease_to_drugs.items()},
        drug_indices_arr,
    )
    test_metrics_off = compute_test_metrics(
        scores_dict=scores_test,
        true_drugs_dict={d: list(v) for d, v in truth_off.items()},
        drug_indices_arr=drug_indices_arr,
        train_pairs=data["train_pairs"],
    )
    _log_test_metrics(wandb.run, "test_ind", test_metrics_ind)
    _log_test_metrics(wandb.run, "test_off", test_metrics_off)
    msg = (
        f"Test (indication-only)  MRR={test_metrics_ind['MRR']:.4f}"
    )
    print(msg, flush=True)
    logger.info(msg)
    msg = (
        f"Test (off-label-aug GT) MRR={test_metrics_off['MRR']:.4f}"
    )
    print(msg, flush=True)
    logger.info(msg)

    # Per-disease CSV
    results_dir = Path(config.get("results_dir", "results/tables"))
    csv_path = results_dir / (
        f"feature_fusion_degree_cond_{encoder_name}_{desc_tier}_{projection}.csv"
    )
    df = _per_disease_csv(
        scores_ind=scores_test, scores_off=scores_test,
        truth_ind={d: set(v) for d, v in test_disease_to_drugs.items()},
        truth_off=truth_off,
        drug_indices_arr=drug_indices_arr,
        disease_to_phenotypes=disease_to_phenotypes,
        out_path=csv_path,
    )

    wandb.finish()
    logger.info(
        f"run_degree_cond_experiment: finished in {_fmt_elapsed(t0)} "
        f"(csv={csv_path})"
    )
    return {
        "phase1": diag,
        "best_val_mrr": finetune_diag["best_val_mrr"],
        "best_epoch": finetune_diag["best_epoch"],
        "epochs_ran": finetune_diag["epochs_ran"],
        "restored_best_state": finetune_diag["restored_best_state"],
        "test_metrics_ind": test_metrics_ind,
        "test_metrics_off": test_metrics_off,
        "fusion_w": float(fusion.w.detach().cpu().item()),
        "fusion_b": float(fusion.b.detach().cpu().item()),
        "log_degree": train_log_degree.detach().cpu(),
        "log_degree_train": train_log_degree.detach().cpu(),
        "log_degree_eval": eval_log_degree.detach().cpu(),
        "drug_indices_arr": drug_indices_arr,
        "per_disease_df": df,
        "csv_path": str(csv_path),
    }


# ── Approach (b): Autoencoder fusion ─────────────────────────────────────
def _phase1_ae_pretraining(
    model: PhenoDrugModel,
    fusion: nn.Module,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    h_llm_full: torch.Tensor,
    has_text_mask: torch.Tensor,
    config: dict,
    wandb_run=None,
) -> list[float]:
    """Unsupervised AE pretraining with reconstruction MSE on text-covered rows.

    Returns the per-epoch reconstruction loss curve.
    """
    import wandb
    t0 = perf_counter()
    epochs = int(config.get("ae_pretrain_epochs", 50))
    lr = float(config.get("ae_pretrain_lr", 1e-3))
    optimizer = torch.optim.Adam(fusion.parameters(), lr=lr)

    # Freeze R-GCN — h_graph computed once
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        h_graph_frozen = model.encode(edge_index, edge_type).detach()

    n_text = int(has_text_mask.sum().item())
    _print_and_log(
        f"Phase 1 AE pretraining: start (epochs={epochs}, lr={lr:.1e}, "
        f"n_rows_in_loss={n_text})"
    )
    if wandb_run is not None:
        wandb.summary["ae_pretrain/n_rows_in_loss"] = n_text

    losses: list[float] = []
    for epoch in range(epochs):
        fusion.train()
        h_g_subset = h_graph_frozen[has_text_mask]
        h_l_subset = h_llm_full[has_text_mask]
        recon_loss = fusion.reconstruction_loss(h_g_subset, h_l_subset)
        optimizer.zero_grad()
        recon_loss.backward()
        optimizer.step()
        losses.append(float(recon_loss.detach().cpu().item()))
        if wandb_run is not None:
            wandb.log({"ae_pretrain/loss": losses[-1], "epoch": epoch})
        if epoch % 10 == 0 or epoch == epochs - 1:
            _print_and_log(
                f"Phase 1 AE pretraining: epoch {epoch + 1}/{epochs} "
                f"loss={losses[-1]:.6f}"
            )
    _print_and_log(
        f"Phase 1 AE pretraining: finished in {_fmt_elapsed(t0)} "
        f"(final_loss={losses[-1]:.6f})"
    )
    return losses


def _run_autoencoder_family_experiment(
    config: dict,
    fusion_variant: str = "ae",
) -> dict:
    """Shared runner for autoencoder-style feature fusion variants."""
    import wandb
    t0 = perf_counter()

    config = dict(config)
    config.setdefault("lr", 1e-4)
    config.setdefault("phase2_train_scope", "scorer_only")
    config.setdefault("phase2_bypass_fusion", False)

    if fusion_variant not in {"ae", "residual_ae"}:
        raise ValueError(
            f"Unsupported fusion_variant={fusion_variant!r}; "
            "expected 'ae' or 'residual_ae'."
        )

    encoder_name = config["encoder_name"]
    desc_tier = config["desc_tier"]
    projection = config["projection"]
    device = torch.device(config.get("device", "cuda"))
    variant_label = "feature-fusion-ae" if fusion_variant == "ae" else "feature-fusion-residual-ae"
    variant_display = "autoencoder" if fusion_variant == "ae" else "residual-autoencoder"

    run_name = f"{variant_label}-{encoder_name}-{desc_tier}-{projection}"
    wandb.init(project="rare-disease-repurposing", name=run_name)
    wandb.config.update(config)
    wandb.summary["fusion_variant"] = fusion_variant
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("ae_pretrain/*", step_metric="epoch")
    _print_and_log(
        f"_run_autoencoder_family_experiment: start variant={fusion_variant}, "
        f"encoder={encoder_name}, tier={desc_tier}, projection={projection}, device={device}"
    )

    data = _setup_data(config, device)
    eval_edge_index, eval_edge_type = data["edge_index"], data["edge_type"]
    drug_indices_arr = data["drug_indices_arr"]
    num_nodes = data["num_nodes"]
    sup = data["supervision"]
    train_disease_to_drugs = sup["train_disease_to_drugs"]
    test_disease_to_drugs = sup["test_disease_to_drugs"]
    disease_to_phenotypes = sup["disease_to_phenotypes"]

    train_subset, val_subset = split_train_val(
        data["train_diseases"],
        val_frac=float(config.get("val_frac", 0.1)),
        seed=int(config.get("seed", 42)),
    )
    train_pairs_subset = data["train_pairs"][
        data["train_pairs"]["disease_id"].isin(train_subset)
    ].reset_index(drop=True)
    train_graph_heldout = set(data["test_diseases"]) | set(val_subset)
    train_edge_index, train_edge_type = _build_runtime_graph(
        data["kg_df"], train_graph_heldout, data["rel2id"], device,
    )
    _print_and_log(
        "_run_autoencoder_family_experiment: rebuilt train_graph after runtime val split "
        f"(heldout_diseases={len(train_graph_heldout)}, "
        f"edge_index_shape={tuple(train_edge_index.shape)})"
    )

    model, _ = build_pheno_drug_model_from_checkpoint(
        config["checkpoint_path"], num_nodes, data["num_relations"], config, device,
    )
    _print_and_log("_run_autoencoder_family_experiment: loaded baseline checkpoint")

    embed_base = Path(config["embed_dir"]) / encoder_name / desc_tier / projection
    drug_path = embed_base / "drug_embeddings.pt"
    pheno_path = embed_base / "phenotype_embeddings.pt"
    if not drug_path.exists() or not pheno_path.exists():
        raise FileNotFoundError(f"Missing embeddings under {embed_base}")
    h_llm_full, has_text_mask = build_h_llm_full(
        drug_path, pheno_path, num_nodes, drug_indices_arr, device,
    )
    assert h_llm_full.shape == (num_nodes, model.node_emb.weight.shape[1]), (
        f"h_llm_full shape {h_llm_full.shape} mismatches model dim "
        f"{model.node_emb.weight.shape[1]}. Cached embeddings must match the "
        "R-GCN hidden_dim — check that embed_dir points at data/embeddings_256."
    )
    _print_and_log(
        f"_run_autoencoder_family_experiment: built h_llm_full "
        f"(shape={tuple(h_llm_full.shape)}, text_rows={int(has_text_mask.sum().item())})"
    )

    embed_dim = int(config.get("embed_dim", 256))
    ae_input_dim = int(config.get("ae_input_dim", 2 * embed_dim))
    ae_hidden_dim = int(config.get("ae_hidden_dim", 2 * embed_dim))
    latent_dim = int(config.get("latent_dim", embed_dim))
    if fusion_variant == "ae":
        fusion: nn.Module = AutoencoderFusion(
            input_dim=ae_input_dim,
            hidden_dim=ae_hidden_dim,
            latent_dim=latent_dim,
        ).to(device)
    else:
        fusion = ResidualAutoencoderFusion(
            input_dim=ae_input_dim,
            hidden_dim=ae_hidden_dim,
            latent_dim=latent_dim,
        ).to(device)
    assert getattr(fusion, "encoder")[0].in_features == 2 * embed_dim, (
        f"Fusion first Linear in_features={getattr(fusion, 'encoder')[0].in_features} "
        f"!= 2*{embed_dim}"
    )
    masked_fusion = MaskedFusionWrapper(fusion, has_text_mask).to(device)
    _print_and_log(
        f"_run_autoencoder_family_experiment: fusion module ready "
        f"(variant={fusion_variant}, input_dim={ae_input_dim}, "
        f"hidden_dim={ae_hidden_dim}, latent_dim={latent_dim})"
    )

    # ── Phase 1: AE pretraining ──
    ae_losses = _phase1_ae_pretraining(
        model=model, fusion=fusion,
        edge_index=train_edge_index, edge_type=train_edge_type,
        h_llm_full=h_llm_full, has_text_mask=has_text_mask,
        config=config, wandb_run=wandb.run,
    )
    # Re-enable gradients on the model + freeze AE decoder for Phase 2
    for p in model.parameters():
        p.requires_grad_(True)
    for p in getattr(fusion, "decoder").parameters():
        p.requires_grad_(False)
    _print_and_log(
        "_run_autoencoder_family_experiment: Phase 1 complete; decoder frozen for Phase 2"
    )

    # ── Phase 2: Supervised fine-tune ──
    finetune_diag = _supervised_finetune(
        model=model, masked_fusion=masked_fusion,
        edge_index=train_edge_index, edge_type=train_edge_type,
        h_llm_full=h_llm_full, has_text_mask=has_text_mask,
        log_degree=None,  # AE fusion does not need log_degree
        train_subset_diseases=train_subset, val_subset_diseases=val_subset,
        train_disease_to_drugs=train_disease_to_drugs,
        disease_to_phenotypes=disease_to_phenotypes,
        drug_indices_arr=drug_indices_arr,
        train_pairs_subset=train_pairs_subset,
        config=config, use_anchoring=False, device=device,
        wandb_run=wandb.run,
        extra_trainable_params=list(fusion.encoder.parameters()),
    )
    wandb.summary["best_val_mrr"] = finetune_diag["best_val_mrr"]
    _print_and_log(
        f"_run_autoencoder_family_experiment: Phase 2 complete "
        f"(best_epoch={finetune_diag['best_epoch']}, "
        f"best_val_mrr={finetune_diag['best_val_mrr']:.4f})"
    )

    # ── Test eval (both GT settings) ──
    _print_and_log("_run_autoencoder_family_experiment: starting test evaluation")
    test_metrics_ind, scores_test = evaluate_fusion_model(
        model=model, masked_fusion=masked_fusion,
        edge_index=eval_edge_index, edge_type=eval_edge_type,
        h_llm_full=h_llm_full, log_degree=None,
        diseases=sorted(data["test_diseases"]),
        disease_to_phenotypes=disease_to_phenotypes,
        disease_to_true_drugs={d: list(v) for d, v in test_disease_to_drugs.items()},
        drug_indices_arr=drug_indices_arr,
        device=device,
        train_pairs=data["train_pairs"],
    )
    truth_off = build_off_label_truth(
        data["kg_df"], data["test_diseases"],
        {d: set(v) for d, v in test_disease_to_drugs.items()},
        drug_indices_arr,
    )
    test_metrics_off = compute_test_metrics(
        scores_dict=scores_test,
        true_drugs_dict={d: list(v) for d, v in truth_off.items()},
        drug_indices_arr=drug_indices_arr,
        train_pairs=data["train_pairs"],
    )
    _log_test_metrics(wandb.run, "test_ind", test_metrics_ind)
    _log_test_metrics(wandb.run, "test_off", test_metrics_off)
    _print_and_log(
        f"_run_autoencoder_family_experiment: test complete "
        f"(MRR_ind={test_metrics_ind['MRR']:.4f}, "
        f"MRR_off={test_metrics_off['MRR']:.4f})"
    )

    results_dir = Path(config.get("results_dir", "results/tables"))
    csv_stem = (
        "feature_fusion_autoencoder"
        if fusion_variant == "ae"
        else "feature_fusion_residual_autoencoder"
    )
    csv_path = results_dir / f"{csv_stem}_{encoder_name}_{desc_tier}_{projection}.csv"
    df = _per_disease_csv(
        scores_ind=scores_test, scores_off=scores_test,
        truth_ind={d: set(v) for d, v in test_disease_to_drugs.items()},
        truth_off=truth_off,
        drug_indices_arr=drug_indices_arr,
        disease_to_phenotypes=disease_to_phenotypes,
        out_path=csv_path,
    )

    wandb.finish()
    _print_and_log(
        f"_run_autoencoder_family_experiment: finished in {_fmt_elapsed(t0)} "
        f"(csv={csv_path})"
    )
    return {
        "fusion_variant": fusion_variant,
        "fusion_display": variant_display,
        "ae_pretrain_losses": ae_losses,
        "best_val_mrr": finetune_diag["best_val_mrr"],
        "best_epoch": finetune_diag["best_epoch"],
        "epochs_ran": finetune_diag["epochs_ran"],
        "test_metrics_ind": test_metrics_ind,
        "test_metrics_off": test_metrics_off,
        "per_disease_df": df,
        "csv_path": str(csv_path),
    }


def run_autoencoder_experiment(config: dict) -> dict:
    """Top-level entry: bottleneck autoencoder fusion."""
    return _run_autoencoder_family_experiment(config, fusion_variant="ae")


def run_residual_autoencoder_experiment(config: dict) -> dict:
    """Top-level entry: residual autoencoder fusion."""
    return _run_autoencoder_family_experiment(config, fusion_variant="residual_ae")
