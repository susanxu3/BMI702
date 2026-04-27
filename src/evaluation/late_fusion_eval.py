"""Late fusion evaluation orchestrator (Phase 2a).

Ties together graph scores + text embedding scores + beta sweep + evaluation.
Designed to be imported from Colab cells.

Usage (in Colab):
    from src.evaluation.late_fusion_eval import run_late_fusion_experiment
    results = run_late_fusion_experiment(config)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.metrics import (
    per_disease_auprc,
    per_disease_auroc,
    recall_at_k,
    reciprocal_rank,
)
from src.models.fusion import LateFusion, normalize_scores
from src.models.train import pad_pheno_batch

logger = logging.getLogger(__name__)

def _extract_state_dict(checkpoint_obj) -> dict:
    """Handle multiple checkpoint formats.

    We expect either a raw state_dict (dict[str, Tensor]) or a wrapper dict
    containing a 'state_dict' key (common in some training scripts).
    """
    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        sd = checkpoint_obj["state_dict"]
        if isinstance(sd, dict):
            return sd
    if isinstance(checkpoint_obj, dict):
        return checkpoint_obj
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint_obj)}")


def _infer_arch_from_state_dict(sd: dict) -> dict[str, int]:
    """Infer PhenoDrugModel architecture params from a state_dict."""
    if "node_emb.weight" not in sd:
        raise KeyError("Checkpoint missing 'node_emb.weight' (not a PhenoDrugModel state_dict?)")

    hidden_dim = int(sd["node_emb.weight"].shape[1])

    conv_idxs = set()
    for k in sd.keys():
        if k.startswith("convs."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                conv_idxs.add(int(parts[1]))
    num_layers = (max(conv_idxs) + 1) if conv_idxs else 0

    num_bases = None
    comp = sd.get("convs.0.comp")
    if comp is not None and hasattr(comp, "shape") and len(comp.shape) == 2:
        num_bases = int(comp.shape[1])

    out = {"hidden_dim": hidden_dim, "num_layers": num_layers}
    if num_bases is not None:
        out["num_bases"] = num_bases
    return out


def _pick_num_heads(hidden_dim: int, requested: int) -> int:
    """Pick a valid num_heads dividing hidden_dim, <= requested, else fallback to 1."""
    if requested <= 0:
        return 1
    for h in range(min(requested, hidden_dim), 0, -1):
        if hidden_dim % h == 0:
            return h
    return 1


# ── Load graph scores ────────────────────────────────────────────────────
def load_graph_scores(
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    diseases: set[int] | list[int],
    disease_to_phenotypes: dict[int, set[int]],
    drug_indices_arr: np.ndarray,
    device: torch.device,
    chunk_size: int = 512,
) -> dict[int, np.ndarray]:
    """Compute graph model scores for all drugs per disease.

    Inference flow (mirrors src/models/train.py evaluate()):
    1. R-GCN forward pass on the masked training subgraph (edge_index,
       edge_type are the same masked tensors used during training — all
       edges incident to test disease nodes already removed). Produces
       embeddings for all 129,375 nodes.
    2. Per disease: gather phenotype node embeddings, pad via pad_pheno_batch().
    3. Chunked drug scoring: iterate all 7,957 drugs in chunks of 512.
       Each chunk uses drug embeddings as queries and phenotype embeddings
       as keys/values in the cross-attention scorer.
    4. Concatenate chunk scores -> (7957,) numpy array = s_graph.

    Args:
        model: Trained PhenoDrugModel.
        edge_index: (2, num_edges) masked training subgraph edges.
        edge_type: (num_edges,) relation types for masked subgraph.
        diseases: Disease node indices to score.
        disease_to_phenotypes: Disease -> set of phenotype node indices.
        drug_indices_arr: Sorted array of all drug node indices (7957,).
        device: Compute device.
        chunk_size: Number of drugs per scoring chunk.

    Returns:
        Dict mapping disease_idx -> (num_drugs,) numpy array of scores.
    """
    model.eval()
    all_drugs_t = torch.tensor(drug_indices_arr, dtype=torch.long, device=device)
    scores_dict: dict[int, np.ndarray] = {}

    with torch.no_grad():
        # Single R-GCN forward pass
        node_embs = model.encode(edge_index, edge_type)

        for disease_idx in diseases:
            phenos = list(disease_to_phenotypes.get(disease_idx, []))
            if not phenos:
                logger.warning(f"Disease {disease_idx} has no phenotypes, skipping")
                continue

            chunk_scores = []
            for c_start in range(0, len(drug_indices_arr), chunk_size):
                c_end = min(c_start + chunk_size, len(drug_indices_arr))
                c_drugs = all_drugs_t[c_start:c_end]
                c_padded, c_mask = pad_pheno_batch(
                    [phenos] * (c_end - c_start), device
                )
                scores = model.score(node_embs, c_drugs, c_padded, c_mask)
                chunk_scores.append(scores.cpu().numpy())

            scores_dict[disease_idx] = np.concatenate(chunk_scores)

    logger.info(f"Computed graph scores for {len(scores_dict)} diseases")
    return scores_dict


# ── Load LLM (text embedding) scores ────────────────────────────────────
def load_llm_scores(
    diseases: set[int] | list[int],
    disease_to_phenotypes: dict[int, set[int]],
    drug_embed_path: str | Path,
    pheno_embed_path: str | Path,
    drug_indices_arr: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Compute text embedding cosine similarity scores per disease.

    For each disease: gather phenotype embeddings -> mean-pool ->
    cosine similarity vs all drug embeddings.

    Args:
        diseases: Disease node indices.
        disease_to_phenotypes: Disease -> set of phenotype node indices.
        drug_embed_path: Path to drug_embeddings.pt file.
        pheno_embed_path: Path to phenotype_embeddings.pt file.
        drug_indices_arr: If provided, asserts that the cached drug embedding
            ordering matches this array (so the returned cosine-sim array can
            be safely fused position-by-position with graph scores).

    Returns:
        Dict mapping disease_idx -> (num_drugs,) numpy array of cosine sims.
    """
    drug_data = torch.load(drug_embed_path, map_location="cpu", weights_only=False)
    pheno_data = torch.load(pheno_embed_path, map_location="cpu", weights_only=False)

    drug_embeds = drug_data["embeddings"]  # (num_drugs, dim)
    drug_indices = drug_data["node_indices"]
    pheno_embeds = pheno_data["embeddings"]  # (num_phenos, dim)
    pheno_indices = pheno_data["node_indices"]

    if drug_indices_arr is not None:
        cached = list(drug_indices)
        expected = drug_indices_arr.tolist()
        if cached != expected:
            mismatch = sum(1 for a, b in zip(cached, expected) if a != b)
            raise ValueError(
                f"Drug embedding ordering does not match drug_indices_arr "
                f"(cached len={len(cached)}, expected len={len(expected)}, "
                f"mismatched positions={mismatch}). Re-run cache_embeddings.py "
                f"or align orderings before fusing."
            )

    # Build phenotype node_index -> row lookup
    pheno_idx_to_row = {idx: row for row, idx in enumerate(pheno_indices)}

    scores_dict: dict[int, np.ndarray] = {}
    n_missing_warned = 0

    for disease_idx in diseases:
        phenos = list(disease_to_phenotypes.get(disease_idx, []))
        if not phenos:
            continue

        # Gather available phenotype embeddings
        rows = []
        missing = 0
        for p_idx in phenos:
            row = pheno_idx_to_row.get(p_idx)
            if row is not None:
                rows.append(row)
            else:
                missing += 1

        if missing > 0 and n_missing_warned < 5:
            logger.warning(
                f"Disease {disease_idx}: {missing}/{len(phenos)} phenotypes "
                f"missing from embedding file"
            )
            n_missing_warned += 1

        if not rows:
            # All phenotypes missing -> uniform scores
            scores_dict[disease_idx] = np.zeros(len(drug_indices), dtype=np.float32)
            continue

        pheno_subset = pheno_embeds[rows]  # (n_available, dim)
        cosine_sims = LateFusion.compute_llm_scores(pheno_subset, drug_embeds)
        scores_dict[disease_idx] = cosine_sims.numpy()

    logger.info(f"Computed LLM scores for {len(scores_dict)} diseases")
    return scores_dict


# ── Cold-start drug stratification ───────────────────────────────────────
def _build_drug_degree(train_pairs: pd.DataFrame) -> dict[int, int]:
    """Count training indication frequency per drug (notebook cell 29 pattern).

    Args:
        train_pairs: DataFrame with 'drug_id' column.

    Returns:
        drug_degree: drug_index -> count of training pairs.
    """
    drug_degree: dict[int, int] = defaultdict(int)
    for _, row in train_pairs.iterrows():
        drug_degree[int(row["drug_id"])] += 1
    return drug_degree


def _stratify_diseases_by_coldstart(
    disease_to_true_drugs: dict[int, list[int]],
    drug_degree: dict[int, int],
    diseases: list[int],
) -> dict[str, list[int]]:
    """Bin diseases into cold-start strata (notebook cell 42 pattern).

    Args:
        disease_to_true_drugs: Disease -> list of true drug indices.
        drug_degree: Drug -> training indication count.
        diseases: Disease indices to stratify.

    Returns:
        {"all": [...], "some": [...], "none": [...]} disease index lists.
    """
    strata: dict[str, list[int]] = {"all": [], "some": [], "none": []}

    for d_idx in diseases:
        true_drugs = disease_to_true_drugs.get(d_idx, [])
        if not true_drugs:
            continue
        n_seen = sum(1 for d in true_drugs if drug_degree.get(d, 0) > 0)
        seen_ratio = n_seen / len(true_drugs)

        if seen_ratio == 0:
            strata["none"].append(d_idx)
        elif seen_ratio < 1.0:
            strata["some"].append(d_idx)
        else:
            strata["all"].append(d_idx)

    return strata


# ── Evaluate a single beta value ─────────────────────────────────────────
def evaluate_single_beta(
    graph_scores: dict[int, np.ndarray],
    llm_scores: dict[int, np.ndarray],
    disease_to_true_drugs: dict[int, list[int]],
    drug_indices_arr: np.ndarray,
    beta: float,
    normalize: str = "minmax",
    train_pairs: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Evaluate fused scores at a given beta on a set of diseases.

    Per-disease evaluation:
    1. Normalize both score arrays independently.
    2. Fuse: s_fused = beta * s_graph_norm + (1-beta) * s_llm_norm.
    3. Rank drugs by descending fused score.
    4. Compute MRR, R@K, full-library per-disease AUROC/AUPRC (macro).
    5. Accumulate 1:1 balanced (label, score) pairs per disease -> pooled micro
       AUROC/AUPRC (TxGNN-style; matches src/pagerank & new model.ipynb).
    6. Cold-start stratification if train_pairs provided.

    Args:
        graph_scores: disease_idx -> (num_drugs,) graph scores.
        llm_scores: disease_idx -> (num_drugs,) text cosine sims.
        disease_to_true_drugs: disease_idx -> list of true drug indices.
        drug_indices_arr: Sorted array of all drug node indices.
        beta: Mixing weight.
        normalize: Normalization method ('minmax' or 'rank').
        train_pairs: If provided, enables cold-start stratification.

    Returns:
        Dict with MRR, R@1/5/10/50, AUROC (full-library macro),
        AUPRC (full-library macro), AUROC_balanced_micro,
        AUPRC_balanced_micro, and optionally MRR_seen_all/_some/_none.
    """
    diseases = sorted(
        set(graph_scores.keys()) & set(llm_scores.keys())
        & set(disease_to_true_drugs.keys())
    )

    rng = np.random.default_rng(42)
    per_disease_results: list[dict] = []
    pooled_labels: list[int] = []
    pooled_scores: list[float] = []

    for d_idx in diseases:
        s_graph = graph_scores[d_idx]
        s_llm = llm_scores[d_idx]
        true_drugs = disease_to_true_drugs[d_idx]

        if not true_drugs:
            continue

        # 1. Normalize
        s_g_norm = normalize_scores(s_graph, method=normalize)
        s_l_norm = normalize_scores(s_llm, method=normalize)

        # 2. Fuse
        s_fused = beta * s_g_norm + (1 - beta) * s_l_norm

        # 3. Rank
        ranked_order = np.argsort(-s_fused)
        ranked_drugs = drug_indices_arr[ranked_order].tolist()
        true_drug_set = set(true_drugs)

        # 4. Metrics (full-library, per-disease)
        mrr = reciprocal_rank(ranked_drugs, true_drugs)
        r1 = recall_at_k(ranked_drugs, true_drugs, 1)
        r5 = recall_at_k(ranked_drugs, true_drugs, 5)
        r10 = recall_at_k(ranked_drugs, true_drugs, 10)
        r50 = recall_at_k(ranked_drugs, true_drugs, 50)
        auroc = per_disease_auroc(s_fused, true_drug_set, drug_indices_arr)
        auprc = per_disease_auprc(s_fused, true_drug_set, drug_indices_arr)

        per_disease_results.append({
            "disease_idx": d_idx,
            "MRR": mrr,
            "R@1": r1,
            "R@5": r5,
            "R@10": r10,
            "R@50": r50,
            "AUROC": auroc,
            "AUPRC": auprc,
        })

        # 5. TxGNN-style pooled 1:1 balanced collection
        pos_mask = np.fromiter(
            (d in true_drug_set for d in drug_indices_arr),
            dtype=bool,
            count=len(drug_indices_arr),
        )
        pos_pos = np.where(pos_mask)[0]
        neg_pos = np.where(~pos_mask)[0]
        n_pos = len(pos_pos)
        if n_pos > 0 and len(neg_pos) >= n_pos:
            sampled_neg = rng.choice(neg_pos, size=n_pos, replace=False)
            pooled_labels.extend([1] * n_pos + [0] * n_pos)
            pooled_scores.extend(
                s_fused[pos_pos].tolist() + s_fused[sampled_neg].tolist()
            )

    if not per_disease_results:
        return {"MRR": 0.0, "R@1": 0.0, "R@5": 0.0, "R@10": 0.0,
                "R@50": 0.0, "AUROC": 0.0, "AUPRC": 0.0,
                "AUROC_balanced_micro": 0.0, "AUPRC_balanced_micro": 0.0}

    # 6. Macro-average (full-library per-disease)
    metrics = {}
    for key in ["MRR", "R@1", "R@5", "R@10", "R@50"]:
        metrics[key] = float(np.mean([r[key] for r in per_disease_results]))

    auroc_vals = [r["AUROC"] for r in per_disease_results if r["AUROC"] is not None]
    auprc_vals = [r["AUPRC"] for r in per_disease_results if r["AUPRC"] is not None]
    metrics["AUROC"] = float(np.mean(auroc_vals)) if auroc_vals else 0.0
    metrics["AUPRC"] = float(np.mean(auprc_vals)) if auprc_vals else 0.0

    # 7. TxGNN-style pooled micro (1:1 balanced, single AP/AUROC over pool)
    if pooled_labels:
        metrics["AUROC_balanced_micro"] = float(
            roc_auc_score(pooled_labels, pooled_scores)
        )
        metrics["AUPRC_balanced_micro"] = float(
            average_precision_score(pooled_labels, pooled_scores)
        )
    else:
        metrics["AUROC_balanced_micro"] = 0.0
        metrics["AUPRC_balanced_micro"] = 0.0

    # 6. Cold-start stratification
    if train_pairs is not None:
        drug_degree = _build_drug_degree(train_pairs)
        disease_list = [r["disease_idx"] for r in per_disease_results]
        disease_to_true = {
            d: disease_to_true_drugs[d] for d in disease_list
        }
        strata = _stratify_diseases_by_coldstart(disease_to_true, drug_degree, disease_list)

        mrr_by_disease = {r["disease_idx"]: r["MRR"] for r in per_disease_results}

        for stratum_name, stratum_diseases in strata.items():
            if stratum_diseases:
                stratum_mrrs = [mrr_by_disease[d] for d in stratum_diseases]
                metrics[f"MRR_seen_{stratum_name}"] = float(np.mean(stratum_mrrs))
            else:
                metrics[f"MRR_seen_{stratum_name}"] = 0.0

        logger.info(
            f"Cold-start strata: all={len(strata['all'])}, "
            f"some={len(strata['some'])}, none={len(strata['none'])}"
        )

    return metrics


# ── Top-level experiment runner ──────────────────────────────────────────
def run_late_fusion_experiment(config: dict) -> dict:
    """Run the full late fusion experiment pipeline.

    Top-level entry point, called from Colab.

    Expected config keys:
        data_dir: Path to PrimeKG data.
        split_dir: Path to train/test split files.
        checkpoint_path: Path to trained R-GCN checkpoint.
        encoder_name: Text encoder name (e.g. 'pubmedbert').
        desc_tier: Description tier ('tier1', 'tier2', 'gpt4o').
        projection: Projection method ('pca', 'linear', 'nonlinear_ae', 'none').
        embed_dir: Base directory for cached embeddings.
        beta_search: List of beta candidates.
        normalize: Normalization method.
        beta_cv_folds: Number of CV folds.
        device: Compute device string.

    Returns:
        Dict with test metrics, beta sweep results, and per-disease DataFrame.
    """
    import wandb

    from src.data.disease_split import load_split
    from src.data.primekg_loader import (
        build_pyg_graph,
        build_supervision_maps,
        load_primekg,
        mask_test_diseases,
    )
    from src.models.cross_attention_scorer import PhenoDrugModel

    device = torch.device(config.get("device", "cuda"))
    encoder_name = config["encoder_name"]
    desc_tier = config["desc_tier"]
    projection = config["projection"]

    # ── wandb init ──
    run_name = f"late-fusion-{encoder_name}-{desc_tier}-{projection}"
    wandb.init(project="rare-disease-repurposing", name=run_name)
    wandb.config.update(config)

    # ── Load data ──
    logger.info("Loading PrimeKG and split...")
    nodes_df, edges_df, kg_df = load_primekg(config["data_dir"])
    train_diseases, test_diseases, train_pairs, test_pairs = load_split(
        config["split_dir"]
    )

    kg_train = mask_test_diseases(kg_df, test_diseases)
    edge_index, edge_type, num_relations, rel2id = build_pyg_graph(kg_train, device)

    supervision = build_supervision_maps(
        kg_df, nodes_df, train_diseases, test_diseases, train_pairs, test_pairs
    )
    disease_to_phenotypes = supervision["disease_to_phenotypes"]
    train_disease_to_drugs = supervision["train_disease_to_drugs"]
    test_disease_to_drugs = supervision["test_disease_to_drugs"]
    drug_indices = supervision["drug_indices"]
    drug_indices_arr = np.array(sorted(drug_indices))

    # ── Load model ──
    logger.info("Loading model checkpoint...")
    checkpoint_obj = torch.load(
        config["checkpoint_path"], map_location=device, weights_only=False
    )
    state_dict = _extract_state_dict(checkpoint_obj)

    num_nodes = len(nodes_df)
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

    # Sanity-check relation count if comp exists.
    comp0 = state_dict.get("convs.0.comp")
    if comp0 is not None and hasattr(comp0, "shape") and len(comp0.shape) == 2:
        ckpt_num_rel = int(comp0.shape[0])
        if ckpt_num_rel != int(num_relations):
            raise ValueError(
                "Checkpoint num_relations mismatch: "
                f"checkpoint has {ckpt_num_rel}, current graph has {num_relations}. "
                "This usually means the relation set/order differs from training."
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

    try:
        model.load_state_dict(state_dict, strict=True)
        print("Model loaded successfully with strict=True")
    except RuntimeError as e:
        # Helpful context for the most common failure: layer-count mismatch.
        logger.error(
            "Failed to load checkpoint with config arch "
            f"(hidden_dim={hidden_dim}, num_layers={num_layers}, num_bases={num_bases}, num_heads={num_heads}). "
            f"Inferred from checkpoint: {inferred}."
        )
        raise

    # ── Compute graph scores (train + test) ──
    all_diseases = list(train_diseases | test_diseases)
    logger.info(f"Computing graph scores for {len(all_diseases)} diseases...")
    graph_scores = load_graph_scores(
        model, edge_index, edge_type,
        all_diseases, disease_to_phenotypes, drug_indices_arr, device,
    )

    # ── Compute LLM scores ──
    embed_base = Path(config["embed_dir"]) / encoder_name / desc_tier / projection
    drug_embed_path = embed_base / "drug_embeddings.pt"
    pheno_embed_path = embed_base / "phenotype_embeddings.pt"

    logger.info(f"Loading text embeddings from {embed_base}")
    llm_scores = load_llm_scores(
        all_diseases, disease_to_phenotypes, drug_embed_path, pheno_embed_path,
        drug_indices_arr=drug_indices_arr,
    )

    # ── Calibrate beta (5-fold CV on train diseases) ──
    train_true_drugs = {
        d: list(drugs) for d, drugs in train_disease_to_drugs.items()
    }
    beta_candidates = config.get(
        "beta_search", [i / 10 for i in range(11)]
    )
    n_folds = config.get("beta_cv_folds", 5)
    norm_method = config.get("normalize", "minmax")

    logger.info(f"Calibrating beta with {n_folds}-fold CV...")
    best_beta, best_cv_mrr, beta_to_mrr = LateFusion.calibrate_beta(
        graph_scores=graph_scores,
        llm_scores=llm_scores,
        disease_to_true_drugs=train_true_drugs,
        drug_indices_arr=drug_indices_arr,
        beta_candidates=beta_candidates,
        normalize=norm_method,
        n_folds=n_folds,
    )

    # Log beta sweep curve
    for b, m in sorted(beta_to_mrr.items()):
        wandb.log({"beta_sweep/beta": b, "beta_sweep/cv_mrr": m})

    logger.info(f"Best beta: {best_beta} (CV MRR: {best_cv_mrr:.4f})")
    wandb.summary["best_beta"] = best_beta
    wandb.summary["best_cv_mrr"] = best_cv_mrr

    # ── Evaluate on 108 test diseases ──
    test_true_drugs = {
        d: list(drugs) for d, drugs in test_disease_to_drugs.items()
    }

    # Best beta
    logger.info(f"Evaluating best beta={best_beta} on test diseases...")
    test_metrics = evaluate_single_beta(
        graph_scores, llm_scores, test_true_drugs, drug_indices_arr,
        beta=best_beta, normalize=norm_method, train_pairs=train_pairs,
    )

    # Ablation baselines: beta=0.0 (pure LLM) and beta=1.0 (pure graph)
    logger.info("Evaluating ablation baselines (beta=0.0, beta=1.0)...")
    pure_llm = evaluate_single_beta(
        graph_scores, llm_scores, test_true_drugs, drug_indices_arr,
        beta=0.0, normalize=norm_method, train_pairs=train_pairs,
    )
    pure_graph = evaluate_single_beta(
        graph_scores, llm_scores, test_true_drugs, drug_indices_arr,
        beta=1.0, normalize=norm_method, train_pairs=train_pairs,
    )

    # ── Log results ──
    for key, val in test_metrics.items():
        wandb.summary[f"test/{key}"] = val
    for key, val in pure_llm.items():
        wandb.summary[f"test_pure_llm/{key}"] = val
    for key, val in pure_graph.items():
        wandb.summary[f"test_pure_graph/{key}"] = val

    logger.info("=== Test Results ===")
    logger.info(f"  Best fusion (beta={best_beta}): MRR={test_metrics['MRR']:.4f}")
    logger.info(f"  Pure graph  (beta=1.0):         MRR={pure_graph['MRR']:.4f}")
    logger.info(f"  Pure LLM    (beta=0.0):         MRR={pure_llm['MRR']:.4f}")

    if "MRR_seen_all" in test_metrics:
        logger.info("  Cold-start stratification:")
        logger.info(f"    MRR (all seen):    {test_metrics['MRR_seen_all']:.4f}")
        logger.info(f"    MRR (some seen):   {test_metrics['MRR_seen_some']:.4f}")
        logger.info(f"    MRR (none seen):   {test_metrics['MRR_seen_none']:.4f}")

    # ── Save per-disease results ──
    results_dir = Path(config.get("results_dir", "results/tables"))
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"late_fusion_{encoder_name}_{desc_tier}_{projection}.csv"

    # Re-run to collect per-disease data for CSV (full-library per-disease metrics)
    per_disease_rows = []
    for d_idx in sorted(test_true_drugs.keys()):
        if d_idx not in graph_scores or d_idx not in llm_scores:
            continue
        true_drugs = test_true_drugs[d_idx]
        if not true_drugs:
            continue

        s_g = normalize_scores(graph_scores[d_idx], norm_method)
        s_l = normalize_scores(llm_scores[d_idx], norm_method)
        s_fused = best_beta * s_g + (1 - best_beta) * s_l

        ranked = drug_indices_arr[np.argsort(-s_fused)].tolist()
        true_set = set(true_drugs)

        per_disease_rows.append({
            "disease_idx": d_idx,
            "n_phenotypes": len(disease_to_phenotypes.get(d_idx, [])),
            "n_true_drugs": len(true_drugs),
            "MRR": reciprocal_rank(ranked, true_drugs),
            "R@10": recall_at_k(ranked, true_drugs, 10),
            "R@50": recall_at_k(ranked, true_drugs, 50),
            "AUROC": per_disease_auroc(s_fused, true_set, drug_indices_arr),
            "AUPRC": per_disease_auprc(s_fused, true_set, drug_indices_arr),
        })

    results_df = pd.DataFrame(per_disease_rows)
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved per-disease results to {results_path}")

    wandb.finish()

    return {
        "best_beta": best_beta,
        "best_cv_mrr": best_cv_mrr,
        "beta_to_mrr": beta_to_mrr,
        "test_metrics": test_metrics,
        "pure_llm_metrics": pure_llm,
        "pure_graph_metrics": pure_graph,
        "per_disease_df": results_df,
    }
