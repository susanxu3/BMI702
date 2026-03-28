from __future__ import annotations

import argparse
import importlib
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
import sys
import zlib

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

# TxGNN still uses the removed pandas.DataFrame.append API in several training
# and evaluation helpers. Restore a small compatibility shim for pandas>=2.
if not hasattr(pd.DataFrame, "append"):
    def _df_append(self, other, ignore_index=False, verify_integrity=False, sort=False):
        if isinstance(other, list):
            pieces = [self, *other]
        else:
            pieces = [self, other]
        return pd.concat(
            pieces,
            ignore_index=ignore_index,
            verify_integrity=verify_integrity,
            sort=sort,
        )

    pd.DataFrame.append = _df_append

try:
    from txgnn import TxEval, TxGNN
    from txgnn.utils import create_dgl_graph, reverse_rel_generation
    txgnn_train_module = importlib.import_module("txgnn.TxGNN")
    txgnn_utils_module = importlib.import_module("txgnn.utils")
    dgl_module = importlib.import_module("dgl")
except ModuleNotFoundError:
    # Colab may have the repository on Drive without an editable install in the
    # current runtime. Fall back to importing from the local repo checkout.
    THIS_DIR = Path(__file__).resolve().parent
    TXGNN_REPO_ROOT = THIS_DIR / "TxGNN"
    if TXGNN_REPO_ROOT.exists() and str(TXGNN_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(TXGNN_REPO_ROOT))
    from txgnn import TxEval, TxGNN
    from txgnn.utils import create_dgl_graph, reverse_rel_generation
    txgnn_train_module = importlib.import_module("txgnn.TxGNN")
    txgnn_utils_module = importlib.import_module("txgnn.utils")
    dgl_module = importlib.import_module("dgl")

# Always source the original implementation from txgnn.utils. The TxGNN module
# may already hold a previously monkeypatched copy across notebook reloads.
_ORIGINAL_EVAL_GRAPH_CONSTRUCT = txgnn_utils_module.evaluate_graph_construct
txgnn_train_module._codex_original_evaluate_graph_construct = _ORIGINAL_EVAL_GRAPH_CONSTRUCT


def _safe_evaluate_graph_construct(df_valid, g, neg_sampler, k, device):
    # DGL 2.5 on Colab can fail when COO->CSR happens during negative-sampler
    # setup on CUDA. Build the eval graphs on CPU, then move them to the target
    # device so the rest of TxGNN can still train/evaluate on GPU.
    g_pos, g_neg = _ORIGINAL_EVAL_GRAPH_CONSTRUCT(df_valid, g, neg_sampler, k, "cpu")
    if str(device) != "cpu":
        g_pos = g_pos.to(device)
        g_neg = g_neg.to(device)
    return g_pos, g_neg


txgnn_train_module.evaluate_graph_construct = _safe_evaluate_graph_construct


DD_RELATIONS = {"indication", "contraindication", "off-label use"}
PHENOTYPE_POS_RELATION = "disease_phenotype_positive"


def convert2str(value) -> str:
    try:
        if "_" not in str(value):
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


@dataclass
class CustomTxDataBundle:
    data_folder: str
    G: object
    df: pd.DataFrame
    df_train: pd.DataFrame
    df_valid: pd.DataFrame
    df_test: pd.DataFrame
    split: str = "custom_disease_eval"
    disease_eval_idx: object = None
    no_kg: bool = False
    seed: int = 42


def _load_pairs(path: Path, left_col: str, right_col: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={left_col: str, right_col: str})
    frame[left_col] = frame[left_col].apply(normalize_node_index)
    frame[right_col] = frame[right_col].apply(normalize_node_index)
    return frame


def _load_id_list(path: Path) -> list[str]:
    with path.open() as handle:
        return [normalize_node_index(line.strip()) for line in handle if line.strip()]


def _load_node_df(path: Path) -> pd.DataFrame:
    node_df = pd.read_csv(path, sep="\t", dtype={"node_index": str, "node_id": str, "node_type": str})
    node_df["node_index"] = node_df["node_index"].apply(normalize_node_index)
    node_df["node_id"] = node_df["node_id"].astype(str).str.strip('"').apply(convert2str)
    node_df["node_type"] = node_df["node_type"].astype(str).str.strip('"')
    return node_df


def _build_id_maps(kg_directed: pd.DataFrame, kg_raw: pd.DataFrame) -> dict:
    kg_directed = kg_directed.copy()
    kg_raw = kg_raw.copy()
    for frame in (kg_directed, kg_raw):
        frame["x_id"] = frame["x_id"].apply(convert2str)
        frame["y_id"] = frame["y_id"].apply(convert2str)

    def collect_idx_map(node_type: str) -> dict[str, int]:
        x_map = kg_directed[kg_directed.x_type == node_type][["x_id", "x_idx"]].drop_duplicates()
        y_map = kg_directed[kg_directed.y_type == node_type][["y_id", "y_idx"]].drop_duplicates()
        merged = pd.concat(
            [
                x_map.rename(columns={"x_id": "entity_id", "x_idx": "entity_idx"}),
                y_map.rename(columns={"y_id": "entity_id", "y_idx": "entity_idx"}),
            ],
            ignore_index=True,
        ).drop_duplicates("entity_id")
        return {convert2str(entity_id): int(entity_idx) for entity_id, entity_idx in merged.values}

    id2idx = {
        "disease": collect_idx_map("disease"),
        "drug": collect_idx_map("drug"),
        "effect/phenotype": collect_idx_map("effect/phenotype"),
    }

    id2name_disease = dict(kg_raw[kg_raw.x_type == "disease"][["x_id", "x_name"]].drop_duplicates().values)
    id2name_disease.update(dict(kg_raw[kg_raw.y_type == "disease"][["y_id", "y_name"]].drop_duplicates().values))

    return {
        "id2idx": id2idx,
        "id2name_disease": {convert2str(k): v for k, v in id2name_disease.items()},
    }


def _derive_validation_diseases(
    train_pairs: pd.DataFrame,
    train_disease_ids: list[str],
    valid_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], pd.DataFrame]:
    if not train_disease_ids:
        raise ValueError("No train diseases were provided.")

    counts = (
        train_pairs.groupby("disease_id")
        .size()
        .rename("num_train_pairs")
        .reset_index()
    )
    counts["num_train_pairs"] = counts["num_train_pairs"].astype(int)
    counts = counts[counts["disease_id"].isin(train_disease_ids)].copy()

    if counts.empty:
        raise ValueError("No train disease IDs matched rows in train_drug_pairs.csv.")

    n_total = len(counts)
    n_valid = max(1, int(round(n_total * valid_ratio)))
    n_valid = min(n_valid, n_total - 1)

    if counts["num_train_pairs"].nunique() > 1 and len(counts) >= 10:
        num_bins = int(min(5, counts["num_train_pairs"].nunique(), len(counts)))
        counts["stratum"] = pd.qcut(
            counts["num_train_pairs"].rank(method="first"),
            q=num_bins,
            labels=False,
            duplicates="drop",
        )
    else:
        counts["stratum"] = 0

    group_sizes = counts.groupby("stratum").size()
    exact = group_sizes / group_sizes.sum() * n_valid
    allocation = np.floor(exact).astype(int)

    remaining = n_valid - int(allocation.sum())
    remainders = (exact - allocation).sort_values(ascending=False)
    for stratum in remainders.index:
        if remaining == 0:
            break
        if allocation[stratum] < group_sizes[stratum]:
            allocation[stratum] += 1
            remaining -= 1

    if remaining > 0:
        for stratum in group_sizes.index:
            if remaining == 0:
                break
            capacity = int(group_sizes[stratum] - allocation[stratum])
            if capacity <= 0:
                continue
            add_now = min(capacity, remaining)
            allocation[stratum] += add_now
            remaining -= add_now

    rng = np.random.default_rng(seed)
    valid_diseases: list[str] = []
    for stratum, group in counts.groupby("stratum", sort=True):
        target = int(allocation.loc[stratum])
        disease_ids = group["disease_id"].tolist()
        rng.shuffle(disease_ids)
        valid_diseases.extend(disease_ids[:target])

    valid_set = set(valid_diseases)
    train_only = [disease_id for disease_id in train_disease_ids if disease_id not in valid_set]

    if not train_only or not valid_set:
        raise ValueError("Validation split derivation failed to leave non-empty train and valid disease sets.")

    counts["split"] = counts["disease_id"].apply(lambda disease_id: "valid" if disease_id in valid_set else "train")
    return train_only, sorted(valid_set), counts.sort_values(["split", "num_train_pairs", "disease_id"]).reset_index(drop=True)


def _make_lookup(node_df: pd.DataFrame, node_type: str) -> dict[str, str]:
    subset = node_df[node_df["node_type"] == node_type][["node_index", "node_id"]].drop_duplicates()
    return dict(zip(subset["node_index"], subset["node_id"]))


def _map_pairs_to_rows(
    pairs: pd.DataFrame,
    node_lookup: dict[str, dict[str, str]],
    id2idx: dict[str, dict[str, int]],
    relation: str,
    left_node_type: str,
    right_node_type: str,
    left_col: str,
    right_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped = pairs.copy()
    mapped["left_raw_id"] = mapped[left_col].map(node_lookup[left_node_type])
    mapped["right_raw_id"] = mapped[right_col].map(node_lookup[right_node_type])
    mapped["left_idx"] = mapped["left_raw_id"].map(id2idx[left_node_type])
    mapped["right_idx"] = mapped["right_raw_id"].map(id2idx[right_node_type])

    good = mapped[
        mapped["left_raw_id"].notna()
        & mapped["right_raw_id"].notna()
        & mapped["left_idx"].notna()
        & mapped["right_idx"].notna()
    ].copy()

    rows = pd.DataFrame(
        {
            "x_type": left_node_type,
            "x_id": good["left_raw_id"].apply(convert2str),
            "relation": relation,
            "y_type": right_node_type,
            "y_id": good["right_raw_id"].apply(convert2str),
            "x_idx": good["left_idx"].astype(int),
            "y_idx": good["right_idx"].astype(int),
        }
    ).drop_duplicates()

    return rows.reset_index(drop=True), mapped


def _prepare_custom_frames(
    project_dir: Path,
    split_dir: Path,
    output_dir: Path,
    valid_ratio: float,
    seed: int,
) -> tuple[dict, dict]:
    data_dir = project_dir / "TxGNN" / "data"
    kg_directed = pd.read_csv(data_dir / "kg_directed.csv", low_memory=False)
    kg_raw = pd.read_csv(data_dir / "kg.csv", low_memory=False)
    node_df = _load_node_df(data_dir / "node.csv")

    for frame in (kg_directed, kg_raw):
        frame["x_id"] = frame["x_id"].apply(convert2str)
        frame["y_id"] = frame["y_id"].apply(convert2str)

    train_pairs = _load_pairs(split_dir / "train_drug_pairs.csv", "disease_id", "drug_id")
    test_pairs = _load_pairs(split_dir / "test_drug_pairs.csv", "disease_id", "drug_id")
    phenotype_pairs = _load_pairs(split_dir / "disease_phenotype_edges.csv", "disease_id", "effect/phenotype_id")
    train_disease_ids = _load_id_list(split_dir / "train_disease_ids.txt")
    test_disease_ids = _load_id_list(split_dir / "test_disease_ids.txt")

    train_only_diseases, valid_diseases, valid_manifest = _derive_validation_diseases(
        train_pairs=train_pairs,
        train_disease_ids=train_disease_ids,
        valid_ratio=valid_ratio,
        seed=seed,
    )

    train_set = set(train_only_diseases)
    valid_set = set(valid_diseases)
    test_set = set(test_disease_ids)
    overlap_train_valid = sorted(train_set & valid_set)
    overlap_train_test = sorted(train_set & test_set)
    overlap_valid_test = sorted(valid_set & test_set)
    if overlap_train_valid or overlap_train_test or overlap_valid_test:
        raise ValueError(
            "Disease split overlap detected: "
            f"train/valid={len(overlap_train_valid)}, "
            f"train/test={len(overlap_train_test)}, "
            f"valid/test={len(overlap_valid_test)}"
        )

    maps = _build_id_maps(kg_directed, kg_raw)
    node_lookup = {
        "disease": _make_lookup(node_df, "disease"),
        "drug": _make_lookup(node_df, "drug"),
        "effect/phenotype": _make_lookup(node_df, "effect/phenotype"),
    }
    valid_raw_ids = {
        node_lookup["disease"][disease_id]
        for disease_id in valid_set
        if disease_id in node_lookup["disease"]
    }
    test_raw_ids = {
        node_lookup["disease"][disease_id]
        for disease_id in test_set
        if disease_id in node_lookup["disease"]
    }

    train_pairs_train = train_pairs[train_pairs["disease_id"].isin(train_set)].reset_index(drop=True)
    train_pairs_valid = train_pairs[train_pairs["disease_id"].isin(valid_set)].reset_index(drop=True)
    test_pairs_only = test_pairs[test_pairs["disease_id"].isin(test_set)].reset_index(drop=True)

    train_indication_rows, mapped_train_pairs = _map_pairs_to_rows(
        pairs=train_pairs_train.rename(columns={"drug_id": "drug_id", "disease_id": "disease_id"}),
        node_lookup=node_lookup,
        id2idx=maps["id2idx"],
        relation="indication",
        left_node_type="drug",
        right_node_type="disease",
        left_col="drug_id",
        right_col="disease_id",
    )
    valid_indication_rows, mapped_valid_pairs = _map_pairs_to_rows(
        pairs=train_pairs_valid.rename(columns={"drug_id": "drug_id", "disease_id": "disease_id"}),
        node_lookup=node_lookup,
        id2idx=maps["id2idx"],
        relation="indication",
        left_node_type="drug",
        right_node_type="disease",
        left_col="drug_id",
        right_col="disease_id",
    )
    test_indication_rows, mapped_test_pairs = _map_pairs_to_rows(
        pairs=test_pairs_only.rename(columns={"drug_id": "drug_id", "disease_id": "disease_id"}),
        node_lookup=node_lookup,
        id2idx=maps["id2idx"],
        relation="indication",
        left_node_type="drug",
        right_node_type="disease",
        left_col="drug_id",
        right_col="disease_id",
    )

    phenotype_rows_all, mapped_phenotypes = _map_pairs_to_rows(
        pairs=phenotype_pairs.rename(columns={"disease_id": "disease_id", "effect/phenotype_id": "effect/phenotype_id"}),
        node_lookup=node_lookup,
        id2idx=maps["id2idx"],
        relation=PHENOTYPE_POS_RELATION,
        left_node_type="disease",
        right_node_type="effect/phenotype",
        left_col="disease_id",
        right_col="effect/phenotype_id",
    )

    phenotype_rows_train = phenotype_rows_all[
        phenotype_rows_all["x_id"].isin([node_lookup["disease"][disease_id] for disease_id in train_set | valid_set if disease_id in node_lookup["disease"]])
    ].copy()
    phenotype_rows_test = phenotype_rows_all[
        phenotype_rows_all["x_id"].isin([node_lookup["disease"][disease_id] for disease_id in test_set if disease_id in node_lookup["disease"]])
    ].copy()

    split_disease_raw_ids = {
        node_lookup["disease"][disease_id]
        for disease_id in train_set | valid_set | test_set
        if disease_id in node_lookup["disease"]
    }

    base_graph_df = kg_directed[kg_directed["relation"] != "indication"].copy()
    base_graph_df = base_graph_df[
        ~(
            (base_graph_df["relation"] == PHENOTYPE_POS_RELATION)
            & (base_graph_df["x_type"] == "disease")
            & (base_graph_df["x_id"].isin(split_disease_raw_ids))
        )
    ].copy()
    aux_dd_rows = kg_directed[kg_directed["relation"].isin(["contraindication", "off-label use"])].copy()
    valid_aux_dd_rows = aux_dd_rows[aux_dd_rows["y_id"].isin(valid_raw_ids)].copy()
    test_aux_dd_rows = aux_dd_rows[aux_dd_rows["y_id"].isin(test_raw_ids)].copy()

    df_train_forward = pd.concat(
        [base_graph_df, phenotype_rows_train, train_indication_rows],
        ignore_index=True,
    ).drop_duplicates(["x_type", "x_id", "relation", "y_type", "y_id"])

    df_eval_forward = pd.concat(
        [df_train_forward, phenotype_rows_test],
        ignore_index=True,
    ).drop_duplicates(["x_type", "x_id", "relation", "y_type", "y_id"])

    df_valid_forward = pd.concat(
        [valid_indication_rows, valid_aux_dd_rows],
        ignore_index=True,
    ).drop_duplicates(["x_type", "x_id", "relation", "y_type", "y_id"])
    df_test_forward = pd.concat(
        [test_indication_rows, test_aux_dd_rows],
        ignore_index=True,
    ).drop_duplicates(["x_type", "x_id", "relation", "y_type", "y_id"])

    unique_rel_train = df_eval_forward[["x_type", "relation", "y_type"]].drop_duplicates()
    df_train = reverse_rel_generation(df_eval_forward, df_train_forward.copy(), unique_rel_train)
    df_valid = reverse_rel_generation(df_eval_forward, df_valid_forward.copy(), unique_rel_train)
    df_test = reverse_rel_generation(df_eval_forward, df_test_forward.copy(), unique_rel_train)
    df_all = reverse_rel_generation(df_eval_forward, df_eval_forward.copy(), unique_rel_train)

    split_artifacts_dir = output_dir / "derived_split"
    split_artifacts_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(split_artifacts_dir / "train.csv", index=False)
    df_valid.to_csv(split_artifacts_dir / "valid.csv", index=False)
    df_test.to_csv(split_artifacts_dir / "test.csv", index=False)
    valid_manifest.to_csv(split_artifacts_dir / "validation_disease_manifest.csv", index=False)
    (split_artifacts_dir / "train_disease_ids.txt").write_text("\n".join(train_only_diseases) + "\n")
    (split_artifacts_dir / "valid_disease_ids.txt").write_text("\n".join(valid_diseases) + "\n")

    mapping_metadata = {
        "train_pairs_total": int(len(train_pairs_train)),
        "train_pairs_mapped": int(len(train_indication_rows)),
        "valid_pairs_total": int(len(train_pairs_valid)),
        "valid_pairs_mapped": int(len(valid_indication_rows)),
        "test_pairs_total": int(len(test_pairs_only)),
        "test_pairs_mapped": int(len(test_indication_rows)),
        "phenotype_edges_total": int(len(phenotype_pairs)),
        "phenotype_edges_mapped": int(len(phenotype_rows_all)),
        "train_diseases_total": int(len(train_only_diseases)),
        "valid_diseases_total": int(len(valid_diseases)),
        "test_diseases_total": int(len(test_disease_ids)),
        "overlap_train_valid": int(len(overlap_train_valid)),
        "overlap_train_test": int(len(overlap_train_test)),
        "overlap_valid_test": int(len(overlap_valid_test)),
    }

    frames = {
        "df_all": df_all,
        "df_train": df_train,
        "df_valid": df_valid,
        "df_test": df_test,
        "df_train_graph_forward": df_train_forward,
        "df_eval_graph_forward": df_eval_forward,
        "maps": maps,
        "mapping_metadata": mapping_metadata,
        "split_artifacts_dir": split_artifacts_dir,
    }
    split_info = {
        "train_diseases": train_only_diseases,
        "valid_diseases": valid_diseases,
        "test_diseases": test_disease_ids,
        "mapped_train_pairs": mapped_train_pairs,
        "mapped_valid_pairs": mapped_valid_pairs,
        "mapped_test_pairs": mapped_test_pairs,
        "mapped_phenotypes": mapped_phenotypes,
    }
    return frames, split_info


def _compute_balanced_metrics(labels: np.ndarray, scores: np.ndarray, repeats: int, seed: int) -> tuple[float, float]:
    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    aurocs = []
    auprcs = []
    sample_size = min(len(pos_idx), len(neg_idx))
    if sample_size == 0:
        return float("nan"), float("nan")

    for _ in range(repeats):
        sampled_neg = rng.choice(neg_idx, size=sample_size, replace=False)
        sampled_pos = rng.choice(pos_idx, size=sample_size, replace=False) if len(pos_idx) > sample_size else pos_idx
        selection = np.concatenate([sampled_pos, sampled_neg])
        labels_bal = labels[selection]
        scores_bal = scores[selection]
        aurocs.append(float(roc_auc_score(labels_bal, scores_bal)))
        auprcs.append(float(average_precision_score(labels_bal, scores_bal)))
    return float(np.mean(auprcs)), float(np.mean(aurocs))


def _run_sparse_safe_disease_eval(model: TxGNN, df_all: pd.DataFrame, df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    relation = "rev_indication"
    train_valid = pd.concat([df_train, df_valid], ignore_index=True)
    df_rel_test = df_test[df_test["relation"] == relation].copy()
    df_rel_train_valid = train_valid[train_valid["relation"] == relation].copy()

    idx2id_drug = dict(df_all[df_all.x_type == "drug"][["x_idx", "x_id"]].drop_duplicates().values)
    idx2id_drug.update(dict(df_all[df_all.y_type == "drug"][["y_idx", "y_id"]].drop_duplicates().values))
    idx2id_disease = dict(df_all[df_all.x_type == "disease"][["x_idx", "x_id"]].drop_duplicates().values)
    idx2id_disease.update(dict(df_all[df_all.y_type == "disease"][["y_idx", "y_id"]].drop_duplicates().values))

    candidate_drug_idxs = np.array(sorted(int(idx) for idx in idx2id_drug.keys()), dtype=np.int64)
    if len(candidate_drug_idxs) == 0:
        raise ValueError("No mapped drug indices available for evaluation.")

    graph_on_device = model.G.to(model.device)
    predictor = model.best_model.eval()
    predictions = {}
    labels_all = {}

    for disease_idx in sorted(df_rel_test["x_idx"].unique().tolist()):
        disease_raw_id = idx2id_disease[int(disease_idx)]
        test_pos = set(df_rel_test[df_rel_test["x_idx"] == disease_idx]["y_idx"].astype(int).tolist())
        train_valid_pos = set(df_rel_train_valid[df_rel_train_valid["x_idx"] == disease_idx]["y_idx"].astype(int).tolist())

        labels = {
            int(drug_idx): 1 if int(drug_idx) in test_pos else (-1 if int(drug_idx) in train_valid_pos else 0)
            for drug_idx in candidate_drug_idxs
        }

        src = torch.full((len(candidate_drug_idxs),), int(disease_idx), dtype=torch.int64, device=model.device)
        dst = torch.as_tensor(candidate_drug_idxs, dtype=torch.int64, device=model.device)
        g_eval = dgl_module.heterograph(
            {("disease", relation, "drug"): (src, dst)},
            num_nodes_dict={ntype: graph_on_device.number_of_nodes(ntype) for ntype in graph_on_device.ntypes},
        ).to(model.device)

        with torch.no_grad():
            _, pred_score_rel, _, _ = predictor(graph_on_device, g_eval)

        scores = pred_score_rel[("disease", relation, "drug")].reshape(-1).detach().cpu().numpy()
        predictions[disease_raw_id] = {
            idx2id_drug[int(drug_idx)]: float(scores[idx]) for idx, drug_idx in enumerate(candidate_drug_idxs)
        }
        labels_all[disease_raw_id] = {
            idx2id_drug[int(drug_idx)]: int(labels[int(drug_idx)]) for drug_idx in candidate_drug_idxs
        }

    return {"prediction": predictions, "label": labels_all}


def _summarize_official_predictions(raw_eval: dict, id2name_disease: dict[str, str], balanced_repeats: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    prediction_dict = raw_eval["prediction"]
    label_dict = raw_eval["label"]

    for disease_id, disease_predictions in prediction_dict.items():
        labels_for_disease = label_dict[disease_id]
        candidate_ids = [drug_id for drug_id, label in labels_for_disease.items() if label != -1]
        scores = np.array([float(disease_predictions[drug_id]) for drug_id in candidate_ids], dtype=float)
        labels = np.array([int(labels_for_disease[drug_id]) for drug_id in candidate_ids], dtype=int)

        pos_idx = np.flatnonzero(labels == 1)
        neg_idx = np.flatnonzero(labels == 0)
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue

        ranked = np.argsort(-scores)
        ranked_labels = labels[ranked]
        hit_positions = np.flatnonzero(ranked_labels == 1)
        auprc_bal, auroc_bal = _compute_balanced_metrics(
            labels=labels,
            scores=scores,
            repeats=balanced_repeats,
            seed=seed + zlib.adler32(str(disease_id).encode("utf-8")),
        )

        rows.append(
            {
                "disease_id": disease_id,
                "disease_name": id2name_disease.get(disease_id, disease_id),
                "candidate_drugs_after_masking": int(len(candidate_ids)),
                "num_masked_train_valid_drugs": int(sum(1 for label in labels_for_disease.values() if label == -1)),
                "num_test_drugs": int(len(pos_idx)),
                "MRR": float(np.mean(1.0 / (hit_positions + 1))),
                "R@1": float(ranked_labels[:1].sum() / len(pos_idx)),
                "R@5": float(ranked_labels[:5].sum() / len(pos_idx)),
                "R@10": float(ranked_labels[:10].sum() / len(pos_idx)),
                "R@50": float(ranked_labels[:50].sum() / len(pos_idx)),
                "AUROC_per_disease": float(roc_auc_score(labels, scores)),
                "AUPRC_balanced_1to1": auprc_bal,
                "AUROC_balanced_1to1": auroc_bal,
            }
        )

    if not rows:
        raise ValueError("No test diseases produced valid evaluation rows after official masking.")

    per_disease_df = pd.DataFrame(rows).sort_values("disease_id").reset_index(drop=True)
    summary_df = pd.DataFrame(
        [
            {"Metric": "MRR", "TxGNN": per_disease_df["MRR"].mean()},
            {"Metric": "R@1", "TxGNN": per_disease_df["R@1"].mean()},
            {"Metric": "R@5", "TxGNN": per_disease_df["R@5"].mean()},
            {"Metric": "R@10", "TxGNN": per_disease_df["R@10"].mean()},
            {"Metric": "R@50", "TxGNN": per_disease_df["R@50"].mean()},
            {"Metric": "AUROC (per-disease mean)", "TxGNN": per_disease_df["AUROC_per_disease"].mean()},
            {"Metric": "AUPRC (balanced 1:1)", "TxGNN": per_disease_df["AUPRC_balanced_1to1"].mean()},
            {"Metric": "AUROC (balanced 1:1)", "TxGNN": per_disease_df["AUROC_balanced_1to1"].mean()},
        ]
    )
    return summary_df, per_disease_df


def _verify_masking(raw_eval: dict, df_train: pd.DataFrame, df_valid: pd.DataFrame) -> dict:
    train_valid = pd.concat([df_train, df_valid], ignore_index=True)
    rel = "rev_indication"
    train_valid_rel = train_valid[train_valid["relation"] == rel]
    expected = {
        convert2str(disease_id): {convert2str(drug_id) for drug_id in group["y_id"].astype(str)}
        for disease_id, group in train_valid_rel.groupby("x_id")
    }

    mismatches = []
    for disease_id, labels in raw_eval["label"].items():
        observed = {convert2str(drug_id) for drug_id, label in labels.items() if label == -1}
        target = expected.get(convert2str(disease_id), set())
        if observed != target:
            mismatches.append(
                {
                    "disease_id": disease_id,
                    "expected_masked": len(target),
                    "observed_masked": len(observed),
                }
            )

    return {
        "masking_mismatch_count": len(mismatches),
        "masking_mismatch_examples": mismatches[:10],
    }


def run_txgnn_leakage_free_eval(
    project_dir: str | Path,
    split_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    seed: int = 42,
    valid_ratio: float = 0.1,
    device: str = "cuda:0",
    pretrain_epochs: int = 1,
    finetune_epochs: int = 500,
    pretrain_lr: float = 1e-3,
    finetune_lr: float = 5e-4,
    batch_size: int = 1024,
    pretrain_print_per_n: int = 20,
    train_print_per_n: int = 5,
    valid_per_n: int = 20,
    balanced_repeats: int = 100,
    use_wandb: bool = False,
    wandb_project: str = "TxGNN_Custom_Split",
    wandb_run_name: str | None = None,
    load_saved_model_dir: str | Path | None = None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    project_dir = Path(project_dir)
    split_dir = Path(split_dir) if split_dir is not None else project_dir / "split"
    output_dir = Path(output_dir) if output_dir is not None else split_dir / "txgnn_leakage_free_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_device = str(device)
    if requested_device.startswith("cuda"):
        try:
            import dgl

            # A CPU-only DGL build raises when graphs are moved to CUDA.
            probe = dgl.graph((torch.tensor([0]), torch.tensor([0])))
            probe = probe.to(requested_device)
            del probe
        except Exception:
            print(
                f"Requested device {requested_device!r}, but the installed DGL build does not support CUDA. "
                "Falling back to CPU."
            )
            device = "cpu"

    required_paths = [
        project_dir / "TxGNN" / "data" / "kg.csv",
        project_dir / "TxGNN" / "data" / "kg_directed.csv",
        project_dir / "TxGNN" / "data" / "node.csv",
        split_dir / "train_drug_pairs.csv",
        split_dir / "test_drug_pairs.csv",
        split_dir / "disease_phenotype_edges.csv",
        split_dir / "train_disease_ids.txt",
        split_dir / "test_disease_ids.txt",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    print("Preparing dataframes and graphs for TxGNN...")
    frames, split_info = _prepare_custom_frames(
        project_dir=project_dir,
        split_dir=split_dir,
        output_dir=output_dir,
        valid_ratio=valid_ratio,
        seed=seed,
    )

    # TxGNN expects the directed graph with reverse relations present in the
    # DGL graph itself, not just in the train/valid/test tables used for
    # masking. Build both graphs from the reverse-augmented dataframes.
    train_graph = create_dgl_graph(frames["df_train"], frames["df_all"])
    eval_graph = create_dgl_graph(frames["df_all"], frames["df_all"])
    data_bundle = CustomTxDataBundle(
        data_folder=str(project_dir / "TxGNN" / "data"),
        G=train_graph,
        df=frames["df_all"],
        df_train=frames["df_train"],
        df_valid=frames["df_valid"],
        df_test=frames["df_test"],
        seed=seed,
    )

    print("Initializing TxGNN model...")
    if use_wandb and not wandb_run_name:
        wandb_run_name = f"TxGNN_Leakage_Free_seed{seed}"
    try:
        model = TxGNN(
            data=data_bundle,
            weight_bias_track=use_wandb,
            proj_name=wandb_project,
            exp_name=wandb_run_name or "TxGNN_Leakage_Free_Custom_Split",
            device=device,
        )
    except Exception as exc:
        if use_wandb:
            raise RuntimeError(
                "Weights & Biases failed to initialize. Reinstall a clean pinned wandb build "
                "in the Colab runtime, restart the session, and rerun the W&B config cell."
            ) from exc
        raise
    model_dir = Path(load_saved_model_dir) if load_saved_model_dir is not None else output_dir / "saved_model"
    if load_saved_model_dir is not None:
        required_model_files = [model_dir / "config.pkl", model_dir / "model.pt"]
        missing_model_files = [str(path) for path in required_model_files if not path.exists()]
        if missing_model_files:
            raise FileNotFoundError(
                "Missing saved-model files for direct evaluation:\n" + "\n".join(missing_model_files)
            )
        print(f"Loading pretrained TxGNN weights from {model_dir}...")
        model.load_pretrained(str(model_dir))
    else:
        model.model_initialize(
            n_hid=100,
            n_inp=100,
            n_out=100,
            proto=True,
            proto_num=3,
            attention=False,
            sim_measure="all_nodes_profile",
            agg_measure="rarity",
        )

        if pretrain_epochs > 0 and hasattr(dgl_module.dataloading, "EdgeDataLoader"):
            model.pretrain(
                n_epoch=pretrain_epochs,
                learning_rate=pretrain_lr,
                batch_size=batch_size,
                train_print_per_n=pretrain_print_per_n,
            )
        elif pretrain_epochs > 0:
            print(
                "Skipping TxGNN pretraining because this DGL build does not expose "
                "dgl.dataloading.EdgeDataLoader. Proceeding directly to finetuning."
            )

        print("Starting TxGNN finetuning...")
        model.finetune(
            n_epoch=finetune_epochs,
            learning_rate=finetune_lr,
            train_print_per_n=train_print_per_n,
            valid_per_n=valid_per_n,
        )

        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_dir))
        print(f"Saved TxGNN model to {model_dir}")

    # Training excludes test-disease phenotype edges. Swap in an augmented graph
    # right before disease-centric evaluation so those phenotypes are available
    # only at inference time. The eval graph still needs the initialized node
    # embeddings stored in G.nodes[ntype].data['inp'], otherwise TxEval fails
    # inside the model forward pass with KeyError('inp').
    for ntype in eval_graph.ntypes:
        if "inp" in model.G.nodes[ntype].data:
            eval_graph.nodes[ntype].data["inp"] = model.G.nodes[ntype].data["inp"].detach().cpu().clone()
    model.G = eval_graph

    print("Evaluating TxGNN on test diseases...")
    raw_eval = _run_sparse_safe_disease_eval(
        model=model,
        df_all=frames["df_all"],
        df_train=frames["df_train"],
        df_valid=frames["df_valid"],
        df_test=frames["df_test"],
    )

    summary_df, per_disease_df = _summarize_official_predictions(
        raw_eval=raw_eval,
        id2name_disease=frames["maps"]["id2name_disease"],
        balanced_repeats=balanced_repeats,
        seed=seed,
    )
    masking_check = _verify_masking(raw_eval=raw_eval, df_train=frames["df_train"], df_valid=frames["df_valid"])

    summary_path = output_dir / "txgnn_leakage_free_summary.csv"
    per_disease_path = output_dir / "txgnn_leakage_free_per_disease.csv"
    raw_eval_path = output_dir / "txgnn_leakage_free_raw_eval.pkl"
    metadata_path = output_dir / "txgnn_leakage_free_metadata.json"
    summary_df.to_csv(summary_path, index=False)
    per_disease_df.to_csv(per_disease_path, index=False)
    with raw_eval_path.open("wb") as handle:
        pickle.dump(raw_eval, handle)

    metadata = {
        "seed": seed,
        "device": device,
        "requested_device": requested_device,
        "valid_ratio": valid_ratio,
        "pretrain_epochs": pretrain_epochs,
        "finetune_epochs": finetune_epochs,
        "pretrain_lr": pretrain_lr,
        "finetune_lr": finetune_lr,
        "batch_size": batch_size,
        "balanced_repeats": balanced_repeats,
        "use_wandb": use_wandb,
        "wandb_project": wandb_project if use_wandb else None,
        "wandb_run_name": wandb_run_name if use_wandb else None,
        "loaded_saved_model_dir": str(model_dir) if load_saved_model_dir is not None else None,
        "results_dir": str(output_dir),
        "summary_path": str(summary_path),
        "per_disease_path": str(per_disease_path),
        "raw_eval_path": str(raw_eval_path),
        "saved_model_dir": str(model_dir),
        "train_diseases": len(split_info["train_diseases"]),
        "valid_diseases": len(split_info["valid_diseases"]),
        "test_diseases": len(split_info["test_diseases"]),
        "mapping": frames["mapping_metadata"],
        "masking_check": masking_check,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return summary_df, per_disease_df, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-free TxGNN training and evaluation on a custom disease split.")
    parser.add_argument("--project-dir", type=str, required=True)
    parser.add_argument("--split-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--finetune-epochs", type=int, default=500)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--pretrain-print-per-n", type=int, default=20)
    parser.add_argument("--train-print-per-n", type=int, default=5)
    parser.add_argument("--valid-per-n", type=int, default=20)
    parser.add_argument("--balanced-repeats", type=int, default=100)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="TxGNN_Custom_Split")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--load-saved-model-dir", type=str, default=None)
    args = parser.parse_args()

    summary_df, _, metadata = run_txgnn_leakage_free_eval(
        project_dir=args.project_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        valid_ratio=args.valid_ratio,
        device=args.device,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        pretrain_lr=args.pretrain_lr,
        finetune_lr=args.finetune_lr,
        batch_size=args.batch_size,
        pretrain_print_per_n=args.pretrain_print_per_n,
        train_print_per_n=args.train_print_per_n,
        valid_per_n=args.valid_per_n,
        balanced_repeats=args.balanced_repeats,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        load_saved_model_dir=args.load_saved_model_dir,
    )
    print(summary_df.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
