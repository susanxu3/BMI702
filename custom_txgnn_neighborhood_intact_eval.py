from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import custom_txgnn_leakage_free_eval as base

from txgnn import TxEval, TxGNN


TARGET_DD_RELATIONS = ("indication", "contraindication")


def _prepare_neighborhood_intact_frames(
    project_dir: Path,
    split_dir: Path,
    output_dir: Path,
    valid_ratio: float,
    seed: int,
) -> tuple[dict, dict]:
    data_dir = project_dir / "TxGNN" / "data"
    kg_directed = pd.read_csv(data_dir / "kg_directed.csv", low_memory=False)
    kg_raw = pd.read_csv(data_dir / "kg.csv", low_memory=False)
    node_df = base._load_node_df(data_dir / "node.csv")

    for frame in (kg_directed, kg_raw):
        frame["x_id"] = frame["x_id"].apply(base.convert2str)
        frame["y_id"] = frame["y_id"].apply(base.convert2str)

    train_pairs = base._load_pairs(split_dir / "train_drug_pairs.csv", "disease_id", "drug_id")
    test_pairs = base._load_pairs(split_dir / "test_drug_pairs.csv", "disease_id", "drug_id")
    train_disease_ids = base._load_id_list(split_dir / "train_disease_ids.txt")
    test_disease_ids = base._load_id_list(split_dir / "test_disease_ids.txt")

    train_only_diseases, valid_diseases, valid_manifest = base._derive_validation_diseases(
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

    maps = base._build_id_maps(kg_directed, kg_raw)
    node_lookup = {
        "disease": base._make_lookup(node_df, "disease"),
        "drug": base._make_lookup(node_df, "drug"),
        "effect/phenotype": base._make_lookup(node_df, "effect/phenotype"),
    }

    train_raw_ids = {
        node_lookup["disease"][disease_id]
        for disease_id in train_set
        if disease_id in node_lookup["disease"]
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
    split_raw_ids = train_raw_ids | valid_raw_ids | test_raw_ids

    train_pairs_train = train_pairs[train_pairs["disease_id"].isin(train_set)].reset_index(drop=True)
    train_pairs_valid = train_pairs[train_pairs["disease_id"].isin(valid_set)].reset_index(drop=True)
    test_pairs_only = test_pairs[test_pairs["disease_id"].isin(test_set)].reset_index(drop=True)

    train_indication_rows, mapped_train_pairs = base._map_pairs_to_rows(
        pairs=train_pairs_train,
        node_lookup=node_lookup,
        id2idx=maps["id2idx"],
        relation="indication",
        left_node_type="drug",
        right_node_type="disease",
        left_col="drug_id",
        right_col="disease_id",
    )
    valid_indication_rows, mapped_valid_pairs = base._map_pairs_to_rows(
        pairs=train_pairs_valid,
        node_lookup=node_lookup,
        id2idx=maps["id2idx"],
        relation="indication",
        left_node_type="drug",
        right_node_type="disease",
        left_col="drug_id",
        right_col="disease_id",
    )
    test_indication_rows, mapped_test_pairs = base._map_pairs_to_rows(
        pairs=test_pairs_only,
        node_lookup=node_lookup,
        id2idx=maps["id2idx"],
        relation="indication",
        left_node_type="drug",
        right_node_type="disease",
        left_col="drug_id",
        right_col="disease_id",
    )

    contraindication_rows = kg_directed[
        (kg_directed["x_type"] == "drug")
        & (kg_directed["relation"] == "contraindication")
        & (kg_directed["y_type"] == "disease")
    ].copy()
    offlabel_rows = kg_directed[
        (kg_directed["x_type"] == "drug")
        & (kg_directed["relation"] == "off-label use")
        & (kg_directed["y_type"] == "disease")
    ].copy()
    train_contra_rows = contraindication_rows[contraindication_rows["y_id"].isin(train_raw_ids)].copy()
    valid_contra_rows = contraindication_rows[contraindication_rows["y_id"].isin(valid_raw_ids)].copy()
    test_contra_rows = contraindication_rows[contraindication_rows["y_id"].isin(test_raw_ids)].copy()
    valid_offlabel_rows = offlabel_rows[offlabel_rows["y_id"].isin(valid_raw_ids)].copy()
    test_offlabel_rows = offlabel_rows[offlabel_rows["y_id"].isin(test_raw_ids)].copy()

    remove_indication_for_split = (
        (kg_directed["x_type"] == "drug")
        & (kg_directed["relation"] == "indication")
        & (kg_directed["y_type"] == "disease")
        & (kg_directed["y_id"].isin(split_raw_ids))
    )
    remove_contra_for_valid_test = (
        (kg_directed["x_type"] == "drug")
        & (kg_directed["relation"] == "contraindication")
        & (kg_directed["y_type"] == "disease")
        & (kg_directed["y_id"].isin(valid_raw_ids | test_raw_ids))
    )
    base_graph_df = kg_directed[~(remove_indication_for_split | remove_contra_for_valid_test)].copy()

    df_train_forward = pd.concat(
        [base_graph_df, train_indication_rows, train_contra_rows],
        ignore_index=True,
    ).drop_duplicates(["x_type", "x_id", "relation", "y_type", "y_id"])

    df_eval_forward = pd.concat(
        [
            df_train_forward,
            valid_indication_rows,
            test_indication_rows,
            valid_contra_rows,
            test_contra_rows,
            valid_offlabel_rows,
            test_offlabel_rows,
        ],
        ignore_index=True,
    ).drop_duplicates(["x_type", "x_id", "relation", "y_type", "y_id"])

    df_valid_forward = pd.concat(
        [valid_indication_rows, valid_contra_rows, valid_offlabel_rows],
        ignore_index=True,
    ).drop_duplicates(["x_type", "x_id", "relation", "y_type", "y_id"])
    df_test_forward = pd.concat(
        [test_indication_rows, test_contra_rows, test_offlabel_rows],
        ignore_index=True,
    ).drop_duplicates(["x_type", "x_id", "relation", "y_type", "y_id"])

    unique_rel = df_eval_forward[["x_type", "relation", "y_type"]].drop_duplicates()
    df_train = base.reverse_rel_generation(df_eval_forward, df_train_forward.copy(), unique_rel)
    df_valid = base.reverse_rel_generation(df_eval_forward, df_valid_forward.copy(), unique_rel)
    df_test = base.reverse_rel_generation(df_eval_forward, df_test_forward.copy(), unique_rel)
    df_all = base.reverse_rel_generation(df_eval_forward, df_eval_forward.copy(), unique_rel)

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
        "train_contraindications": int(len(train_contra_rows)),
        "valid_contraindications": int(len(valid_contra_rows)),
        "test_contraindications": int(len(test_contra_rows)),
        "valid_offlabel": int(len(valid_offlabel_rows)),
        "test_offlabel": int(len(test_offlabel_rows)),
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
    }
    return frames, split_info


def _test_disease_idxs(frames: dict) -> list[int]:
    df_rel_test = frames["df_test"][frames["df_test"]["relation"] == "rev_indication"]
    return sorted(df_rel_test["x_idx"].astype(int).unique().tolist())


def _run_eval_with_official_fallback(
    model: TxGNN,
    frames: dict,
) -> tuple[dict, str]:
    disease_idxs = _test_disease_idxs(frames)
    if not disease_idxs:
        raise ValueError("No held-out test diseases with rev_indication edges were found for evaluation.")

    try:
        official_eval = TxEval(model=model).eval_disease_centric(
            disease_idxs=disease_idxs,
            relation="indication",
            save_result=False,
            show_plot=False,
            verbose=False,
            return_raw=True,
            simulate_random=False,
        )
        masking_check = base._verify_masking(
            raw_eval=official_eval,
            df_train=frames["df_train"],
            df_valid=frames["df_valid"],
        )
        if masking_check["masking_mismatch_count"] == 0:
            return official_eval, "official_txeval"
    except Exception:
        pass

    raw_eval = base._run_sparse_safe_disease_eval(
        model=model,
        df_all=frames["df_all"],
        df_train=frames["df_train"],
        df_valid=frames["df_valid"],
        df_test=frames["df_test"],
    )
    return raw_eval, "custom_sparse_safe"


def run_txgnn_neighborhood_intact_eval(
    project_dir: str | Path,
    split_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    seed: int = 42,
    valid_ratio: float = 0.1,
    device: str = "cuda:0",
    pretrain_epochs: int = 2,
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
    base.random.seed(seed)
    base.np.random.seed(seed)
    torch.manual_seed(seed)

    project_dir = Path(project_dir)
    split_dir = Path(split_dir) if split_dir is not None else project_dir / "split"
    output_dir = Path(output_dir) if output_dir is not None else split_dir / "txgnn_neighborhood_intact_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_device = str(device)
    if requested_device.startswith("cuda"):
        try:
            probe = base.dgl_module.graph((torch.tensor([0]), torch.tensor([0])))
            probe = probe.to(requested_device)
            del probe
        except Exception:
            print(
                f"Requested device {requested_device!r}, but the installed DGL build does not support CUDA. "
                "Falling back to CPU."
            )
            device = "cpu"

    print("Preparing neighborhood-intact dataframes and graphs for TxGNN...")
    frames, split_info = _prepare_neighborhood_intact_frames(
        project_dir=project_dir,
        split_dir=split_dir,
        output_dir=output_dir,
        valid_ratio=valid_ratio,
        seed=seed,
    )

    train_graph = base.create_dgl_graph(frames["df_train"], frames["df_all"])
    data_bundle = base.CustomTxDataBundle(
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
        wandb_run_name = f"TxGNN_Neighborhood_Intact_seed{seed}"

    model = TxGNN(
        data=data_bundle,
        weight_bias_track=use_wandb,
        proj_name=wandb_project,
        exp_name=wandb_run_name or "TxGNN_Neighborhood_Intact_Custom_Split",
        device=device,
    )

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

        if pretrain_epochs > 0 and hasattr(base.dgl_module.dataloading, "EdgeDataLoader"):
            print("Starting TxGNN pretraining...")
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

    print("Evaluating TxGNN on test diseases...")
    raw_eval, eval_backend = _run_eval_with_official_fallback(model=model, frames=frames)
    summary_df, per_disease_df = base._summarize_official_predictions(
        raw_eval=raw_eval,
        id2name_disease=frames["maps"]["id2name_disease"],
        balanced_repeats=balanced_repeats,
        seed=seed,
    )
    masking_check = base._verify_masking(raw_eval=raw_eval, df_train=frames["df_train"], df_valid=frames["df_valid"])

    summary_path = output_dir / "txgnn_neighborhood_intact_summary.csv"
    per_disease_path = output_dir / "txgnn_neighborhood_intact_per_disease.csv"
    raw_eval_path = output_dir / "txgnn_neighborhood_intact_raw_eval.pkl"
    metadata_path = output_dir / "txgnn_neighborhood_intact_metadata.json"
    summary_df.to_csv(summary_path, index=False)
    per_disease_df.to_csv(per_disease_path, index=False)
    with raw_eval_path.open("wb") as handle:
        base.pickle.dump(raw_eval, handle)

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
        "edge_policy": "remove_test_and_valid_indication_contraindication_keep_neighborhood_intact",
        "removed_relations": [
            "indication",
            "rev_indication",
            "contraindication",
            "rev_contraindication",
        ],
        "retained_relations": ["off-label use", "rev_off-label use", "disease_phenotype_positive"],
        "eval_backend": eval_backend,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return summary_df, per_disease_df, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TxGNN training/evaluation on a custom split while keeping phenotype and neighborhood edges intact."
    )
    parser.add_argument("--project-dir", type=str, required=True)
    parser.add_argument("--split-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pretrain-epochs", type=int, default=0)
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

    summary_df, _, metadata = run_txgnn_neighborhood_intact_eval(
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
