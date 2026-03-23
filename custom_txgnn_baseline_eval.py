from __future__ import annotations

import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def convert2str(value) -> str:
    try:
        if "_" in str(value):
            pass
        else:
            value = float(value)
    except Exception:
        pass
    return str(value)


def normalize_node_index(value) -> str:
    value = str(value).strip().strip('"')
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except Exception:
        pass
    return value


def distmult_score(h_src: torch.Tensor, h_rel: torch.Tensor, h_dst: torch.Tensor) -> torch.Tensor:
    # This is the paper's DistMult-style decoder applied directly to cached node
    # embeddings. It avoids rebuilding the DGL graph and avoids running the full
    # TxGNN forward pass used in the official disease-centric evaluator.
    if h_src.dim() == 1:
        h_src = h_src.unsqueeze(0)
    src_rel = h_src * h_rel.unsqueeze(0)
    return torch.sigmoid(torch.mm(src_rel, h_dst.t())).squeeze(0)


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _build_id_mappings(kg_directed: pd.DataFrame, kg_raw: pd.DataFrame) -> dict:
    for frame in (kg_directed, kg_raw):
        frame["x_id"] = frame["x_id"].apply(convert2str)
        frame["y_id"] = frame["y_id"].apply(convert2str)

    idx2id_disease = dict(kg_directed[kg_directed.x_type == "disease"][["x_idx", "x_id"]].drop_duplicates().values)
    idx2id_disease.update(
        dict(kg_directed[kg_directed.y_type == "disease"][["y_idx", "y_id"]].drop_duplicates().values)
    )
    idx2id_drug = dict(kg_directed[kg_directed.x_type == "drug"][["x_idx", "x_id"]].drop_duplicates().values)
    idx2id_drug.update(
        dict(kg_directed[kg_directed.y_type == "drug"][["y_idx", "y_id"]].drop_duplicates().values)
    )
    idx2id_pheno = dict(
        kg_directed[kg_directed.x_type == "effect/phenotype"][["x_idx", "x_id"]].drop_duplicates().values
    )
    idx2id_pheno.update(
        dict(kg_directed[kg_directed.y_type == "effect/phenotype"][["y_idx", "y_id"]].drop_duplicates().values)
    )

    id2idx_disease = {convert2str(entity_id): int(idx) for idx, entity_id in idx2id_disease.items()}
    id2idx_drug = {convert2str(entity_id): int(idx) for idx, entity_id in idx2id_drug.items()}
    id2idx_pheno = {convert2str(entity_id): int(idx) for idx, entity_id in idx2id_pheno.items()}

    id2name_disease = dict(kg_raw[kg_raw.x_type == "disease"][["x_id", "x_name"]].drop_duplicates().values)
    id2name_disease.update(dict(kg_raw[kg_raw.y_type == "disease"][["y_id", "y_name"]].drop_duplicates().values))

    id2name_drug = dict(kg_raw[kg_raw.x_type == "drug"][["x_id", "x_name"]].drop_duplicates().values)
    id2name_drug.update(dict(kg_raw[kg_raw.y_type == "drug"][["y_id", "y_name"]].drop_duplicates().values))

    return {
        "idx2id_disease": {int(idx): convert2str(entity_id) for idx, entity_id in idx2id_disease.items()},
        "id2idx_disease": id2idx_disease,
        "id2idx_drug": id2idx_drug,
        "id2idx_pheno": id2idx_pheno,
        "id2name_disease": id2name_disease,
        "id2name_drug": id2name_drug,
    }


def _extract_indication_weight(kg_directed: pd.DataFrame, state_dict: dict) -> torch.Tensor:
    # The released explorer checkpoint stores one relation vector per edge type.
    # We recover the "drug --indication--> disease" weight and use it as the
    # baseline scorer for the custom split.
    unique_etypes = kg_directed[["x_type", "relation", "y_type"]].drop_duplicates()
    canonical_etypes = [tuple(row) for _, row in unique_etypes.iterrows()]
    rel_name_to_triple = {etype[1]: etype for etype in canonical_etypes}
    sorted_rel_names = sorted(rel_name_to_triple.keys())
    rel2idx = {rel_name_to_triple[rel_name]: idx for idx, rel_name in enumerate(sorted_rel_names)}
    indication_idx = rel2idx[("drug", "indication", "disease")]
    return state_dict["pred.W"][indication_idx]


def _load_split_pairs(path: Path, left_col: str, right_col: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={left_col: str, right_col: str})
    frame[left_col] = frame[left_col].apply(normalize_node_index)
    frame[right_col] = frame[right_col].apply(normalize_node_index)
    return frame


def _load_id_list(path: Path) -> list[str]:
    with path.open() as handle:
        return [normalize_node_index(line.strip()) for line in handle if line.strip()]


def _load_node_index_lookup(node_path: Path) -> dict:
    node_df = pd.read_csv(node_path, sep="\t", dtype={"node_index": str, "node_id": str, "node_type": str})
    node_df["node_index"] = node_df["node_index"].apply(normalize_node_index)
    node_df["node_id"] = node_df["node_id"].astype(str).str.strip('"').apply(convert2str)
    node_df["node_type"] = node_df["node_type"].astype(str).str.strip('"')

    lookup = {}
    for node_type in ["disease", "drug", "effect/phenotype"]:
        subset = node_df[node_df["node_type"] == node_type]
        lookup[node_type] = dict(zip(subset["node_index"], subset["node_id"]))
    return lookup


def _evaluate_single_disease(
    scores: np.ndarray,
    positive_drugs: set[int],
    masked_drugs: set[int],
    balanced_repeats: int,
    seed: int,
) -> dict | None:
    '''
    Evaluation here is ranking over the full drug library after masking custom
    train positives for the same disease. This is simpler than the official
    TxEval disease-centric evaluation in TxGNN, which operates on a model
    trained on a prepared split and can mask all known train/valid relations.
    '''
    all_drug_indices = np.arange(len(scores))
    valid_mask = np.ones(len(scores), dtype=bool)
    if masked_drugs:
        valid_mask[list(masked_drugs)] = False

    candidate_indices = all_drug_indices[valid_mask]
    labels = np.isin(candidate_indices, list(positive_drugs)).astype(int)
    scores_valid = scores[valid_mask]

    num_pos = int(labels.sum())
    num_neg = int((labels == 0).sum())
    if num_pos == 0 or num_neg == 0:
        return None

    ranked = np.argsort(-scores_valid)
    ranked_labels = labels[ranked]

    metrics = {
        "R@1": float(ranked_labels[:1].sum() / num_pos),
        "R@5": float(ranked_labels[:5].sum() / num_pos),
        "R@10": float(ranked_labels[:10].sum() / num_pos),
        "R@50": float(ranked_labels[:50].sum() / num_pos),
        "MRR": float(np.mean(1.0 / (np.flatnonzero(ranked_labels == 1) + 1))),
        "AUROC_per_disease": float(roc_auc_score(labels, scores_valid)),
    }

    pos_positions = np.where(labels == 1)[0]
    neg_positions = np.where(labels == 0)[0]
    rng = np.random.default_rng(seed)

    balanced_aurocs = []
    balanced_auprcs = []
    # The balanced 1:1 AUROC/AUPRC are a convenience metric for this project.
    # They are not the paper's main full-library disease-centric ranking metric.
    for _ in range(balanced_repeats):
        sampled_neg = rng.choice(neg_positions, size=len(pos_positions), replace=False)
        selection = np.concatenate([pos_positions, sampled_neg])
        labels_bal = labels[selection]
        scores_bal = scores_valid[selection]
        balanced_aurocs.append(float(roc_auc_score(labels_bal, scores_bal)))
        balanced_auprcs.append(float(average_precision_score(labels_bal, scores_bal)))

    metrics["AUROC_balanced_1to1"] = float(np.mean(balanced_aurocs))
    metrics["AUPRC_balanced_1to1"] = float(np.mean(balanced_auprcs))
    metrics["num_pos"] = num_pos
    metrics["num_neg"] = num_neg
    return metrics


def run_txgnn_custom_split_baseline(
    project_dir: str | Path,
    split_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    balanced_repeats: int = 100,
    seed: int = 42,
):
    """
    Evaluate an inference-only TxGNN baseline on the custom split folder.

    Current pipeline:
    1. Load cached TxGNNExplorer embeddings and checkpoint weights.
    2. Map custom split PrimeKG node_index values through node.csv.
    3. Score each test disease against all drugs with the cached indication weight.
    4. Compute ranking metrics plus a balanced 1:1 AUROC/AUPRC summary.

    Important: this is not the same as the original paper's full evaluation
    protocol. The paper's core results use TxGNN models trained or fine-tuned on
    a held-out split and evaluated with the official disease-centric evaluator.
    Here we reuse the released explorer checkpoint as a lightweight baseline on a
    custom split, which is faster but can differ methodologically.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    project_dir = Path(project_dir)
    split_dir = Path(split_dir) if split_dir is not None else project_dir / "split"
    results_dir = Path(results_dir) if results_dir is not None else split_dir / "txgnn_baseline_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    explorer_dir = project_dir / "TxGNN" / "TxGNNExplorer"
    data_dir = project_dir / "TxGNN" / "data"

    required_paths = [
        explorer_dir / "node_emb.pkl",
        explorer_dir / "model.pt",
        data_dir / "node.csv",
        data_dir / "kg.csv",
        data_dir / "kg_directed.csv",
        split_dir / "train_drug_pairs.csv",
        split_dir / "test_drug_pairs.csv",
        split_dir / "disease_phenotype_edges.csv",
        split_dir / "train_disease_ids.txt",
        split_dir / "test_disease_ids.txt",
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing_paths))

    # Reusing the released explorer checkpoint is convenient for a quick
    # baseline, but it is not equivalent to retraining TxGNN on the custom split.
    # If the checkpoint already contains signal from the held-out pairs, the
    # resulting numbers will be optimistic relative to a clean paper-style split.
    node_emb = _load_pickle(explorer_dir / "node_emb.pkl")
    state_dict = torch.load(explorer_dir / "model.pt", map_location="cpu")
    node_index_lookup = _load_node_index_lookup(data_dir / "node.csv")

    kg_raw = pd.read_csv(data_dir / "kg.csv", low_memory=False)
    kg_directed = pd.read_csv(data_dir / "kg_directed.csv", low_memory=False)
    mappings = _build_id_mappings(kg_directed, kg_raw)
    indication_weight = _extract_indication_weight(kg_directed, state_dict)

    disease_emb = node_emb["disease"]
    drug_emb = node_emb["drug"]

    train_pairs = _load_split_pairs(split_dir / "train_drug_pairs.csv", "disease_id", "drug_id")
    test_pairs = _load_split_pairs(split_dir / "test_drug_pairs.csv", "disease_id", "drug_id")
    phenotype_pairs = _load_split_pairs(split_dir / "disease_phenotype_edges.csv", "disease_id", "effect/phenotype_id")
    train_disease_ids = set(_load_id_list(split_dir / "train_disease_ids.txt"))
    test_disease_ids = set(_load_id_list(split_dir / "test_disease_ids.txt"))

    idx2id_disease = mappings["idx2id_disease"]
    id2idx_disease = mappings["id2idx_disease"]
    id2idx_drug = mappings["id2idx_drug"]
    id2idx_pheno = mappings["id2idx_pheno"]
    id2name_disease = mappings["id2name_disease"]

    # The split CSVs store PrimeKG global node_index values, while TxGNN scoring
    # uses type-specific x_idx/y_idx plus raw entity IDs. We bridge those two ID
    # systems here before touching embeddings.
    train_pairs["disease_raw_id"] = train_pairs["disease_id"].map(node_index_lookup["disease"])
    train_pairs["drug_raw_id"] = train_pairs["drug_id"].map(node_index_lookup["drug"])
    test_pairs["disease_raw_id"] = test_pairs["disease_id"].map(node_index_lookup["disease"])
    test_pairs["drug_raw_id"] = test_pairs["drug_id"].map(node_index_lookup["drug"])
    phenotype_pairs["disease_raw_id"] = phenotype_pairs["disease_id"].map(node_index_lookup["disease"])
    phenotype_pairs["pheno_raw_id"] = phenotype_pairs["effect/phenotype_id"].map(node_index_lookup["effect/phenotype"])

    mapped_train = train_pairs[
        train_pairs["disease_raw_id"].isin(id2idx_disease) & train_pairs["drug_raw_id"].isin(id2idx_drug)
    ].copy()
    mapped_test = test_pairs[
        test_pairs["disease_raw_id"].isin(id2idx_disease) & test_pairs["drug_raw_id"].isin(id2idx_drug)
    ].copy()
    mapped_pheno = phenotype_pairs[
        phenotype_pairs["disease_raw_id"].isin(id2idx_disease) & phenotype_pairs["pheno_raw_id"].isin(id2idx_pheno)
    ].copy()

    if mapped_test.empty:
        raise ValueError(
            "No test drug pairs mapped from split node indices to TxGNN graph indices. "
            "Check that the split files and PrimeKG node.csv come from the same graph build."
        )

    # Only custom train drug pairs are masked here. This is another difference
    # from the official TxGNN evaluation code, which can mask broader train/valid
    # drug-disease knowledge depending on the prepared split.
    train_by_disease: dict[int, set[int]] = defaultdict(set)
    for row in mapped_train.itertuples(index=False):
        train_by_disease[id2idx_disease[row.disease_raw_id]].add(id2idx_drug[row.drug_raw_id])

    test_by_disease: dict[int, set[int]] = defaultdict(set)
    for row in mapped_test.itertuples(index=False):
        test_by_disease[id2idx_disease[row.disease_raw_id]].add(id2idx_drug[row.drug_raw_id])

    pheno_by_disease: dict[int, set[int]] = defaultdict(set)
    for row in mapped_pheno.to_dict("records"):
        pheno_by_disease[id2idx_disease[row["disease_raw_id"]]].add(id2idx_pheno[row["pheno_raw_id"]])

    target_test_raw_ids = [
        node_index_lookup["disease"][disease_index]
        for disease_index in test_disease_ids
        if disease_index in node_index_lookup["disease"]
    ]
    target_test_indices = [id2idx_disease[disease_id] for disease_id in target_test_raw_ids if disease_id in id2idx_disease]

    per_disease_rows = []
    for disease_idx in sorted(target_test_indices):
        positive_drugs = test_by_disease.get(disease_idx, set())
        if not positive_drugs:
            continue

        disease_vector = disease_emb[disease_idx]
        scores = distmult_score(disease_vector, indication_weight, drug_emb).detach().cpu().numpy()
        metrics = _evaluate_single_disease(
            scores=scores,
            positive_drugs=positive_drugs,
            masked_drugs=train_by_disease.get(disease_idx, set()),
            balanced_repeats=balanced_repeats,
            seed=seed + int(disease_idx),
        )
        if metrics is None:
            continue

        disease_id = idx2id_disease[disease_idx]
        per_disease_rows.append(
            {
                "disease_idx": disease_idx,
                "disease_id": disease_id,
                "disease_name": id2name_disease.get(disease_id, disease_id),
                "num_train_drugs_masked": len(train_by_disease.get(disease_idx, set())),
                "num_test_drugs": len(positive_drugs),
                "num_phenotypes": len(pheno_by_disease.get(disease_idx, set())),
                **metrics,
            }
        )

    per_disease_df = pd.DataFrame(per_disease_rows)
    if not per_disease_df.empty:
        per_disease_df = per_disease_df.sort_values("disease_idx").reset_index(drop=True)
    else:
        raise ValueError(
            "No diseases were evaluated. "
            f"Mapped test diseases: {len(target_test_indices)}; "
            f"mapped test pairs: {len(mapped_test)}; "
            f"mapped train pairs: {len(mapped_train)}."
        )

    summary_df = pd.DataFrame(
        [
            {"Metric": "MRR", "TxGNN Baseline": per_disease_df["MRR"].mean()},
            {"Metric": "R@1", "TxGNN Baseline": per_disease_df["R@1"].mean()},
            {"Metric": "R@5", "TxGNN Baseline": per_disease_df["R@5"].mean()},
            {"Metric": "R@10", "TxGNN Baseline": per_disease_df["R@10"].mean()},
            {"Metric": "R@50", "TxGNN Baseline": per_disease_df["R@50"].mean()},
            {"Metric": "AUROC (per-disease mean)", "TxGNN Baseline": per_disease_df["AUROC_per_disease"].mean()},
            {"Metric": "AUPRC (balanced 1:1)", "TxGNN Baseline": per_disease_df["AUPRC_balanced_1to1"].mean()},
            {"Metric": "AUROC (balanced 1:1)", "TxGNN Baseline": per_disease_df["AUROC_balanced_1to1"].mean()},
        ]
    )

    summary_df.to_csv(results_dir / "txgnn_baseline_summary.csv", index=False)
    per_disease_df.to_csv(results_dir / "txgnn_baseline_per_disease.csv", index=False)

    metadata = {
        "num_test_diseases_listed": len(test_disease_ids),
        "num_test_diseases_mapped": len(target_test_indices),
        "num_test_diseases_evaluated": len(per_disease_df),
        "num_train_diseases_listed": len(train_disease_ids),
        "num_train_diseases_mapped": len(
            [
                d
                for d in train_disease_ids
                if d in node_index_lookup["disease"] and node_index_lookup["disease"][d] in id2idx_disease
            ]
        ),
        "num_train_pairs_total": len(train_pairs),
        "num_train_pairs_mapped": len(mapped_train),
        "num_test_pairs_total": len(test_pairs),
        "num_test_pairs_mapped": len(mapped_test),
        "num_phenotype_edges_total": len(phenotype_pairs),
        "num_phenotype_edges_mapped": len(mapped_pheno),
        "balanced_repeats": balanced_repeats,
        "results_dir": str(results_dir),
    }

    return summary_df, per_disease_df, metadata
