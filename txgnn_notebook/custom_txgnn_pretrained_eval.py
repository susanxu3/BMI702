from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import custom_txgnn_leakage_free_eval as base

from txgnn import TxGNN
from txgnn.utils import Minibatch_NegSampler, get_all_metrics_fb, get_n_params


def _pretrain_with_dgl_compat(
    model: TxGNN,
    n_epoch: int,
    learning_rate: float,
    batch_size: int,
    train_print_per_n: int,
) -> None:
    if model.no_kg:
        raise ValueError("During No-KG ablation, pretraining is infeasible because it is the same as finetuning...")

    dgl = base.dgl_module
    model.G = model.G.to("cpu")
    print("Creating minibatch pretraining dataloader...")

    train_eid_dict = {etype: model.G.edges(form="eid", etype=etype) for etype in model.G.canonical_etypes}
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

    if hasattr(dgl.dataloading, "EdgeDataLoader"):
        dataloader = dgl.dataloading.EdgeDataLoader(
            model.G,
            train_eid_dict,
            sampler,
            negative_sampler=Minibatch_NegSampler(model.G, 1, "fix_dst"),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
    else:
        edge_sampler = dgl.dataloading.as_edge_prediction_sampler(
            sampler,
            negative_sampler=Minibatch_NegSampler(model.G, 1, "fix_dst"),
        )
        dataloader = dgl.dataloading.DataLoader(
            model.G,
            train_eid_dict,
            edge_sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

    optimizer = torch.optim.AdamW(model.model.parameters(), lr=learning_rate)
    print("Start pre-training with #param: %d" % get_n_params(model.model))

    for epoch in range(n_epoch):
        for step, batch in enumerate(dataloader):
            nodes, pos_g, neg_g, blocks = batch
            del nodes
            blocks = [block.to(model.device) for block in blocks]
            pos_g = pos_g.to(model.device)
            neg_g = neg_g.to(model.device)
            pred_score_pos, pred_score_neg, pos_score, neg_score = model.model.forward_minibatch(
                pos_g, neg_g, blocks, model.G, mode="train", pretrain_mode=True
            )

            scores = torch.cat((pos_score, neg_score)).reshape(-1)
            labels = [1] * len(pos_score) + [0] * len(neg_score)

            loss = F.binary_cross_entropy(scores, torch.tensor(labels, dtype=torch.float32, device=model.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if model.weight_bias_track:
                model.wandb.log({"Pretraining Loss": loss})

            if step % train_print_per_n == 0:
                auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc = get_all_metrics_fb(
                    pred_score_pos,
                    pred_score_neg,
                    scores.reshape(-1).detach().cpu().numpy(),
                    labels,
                    model.G,
                    True,
                )
                if model.weight_bias_track:
                    temp_d = base.txgnn_utils_module.get_wandb_log_dict(
                        auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc, "Pretraining"
                    )
                    temp_d.update({"Pretraining LR": optimizer.param_groups[0]["lr"]})
                    model.wandb.log(temp_d)

                print(
                    "Epoch: %d Step: %d LR: %.5f Loss %.4f, Pretrain Micro AUROC %.4f "
                    "Pretrain Micro AUPRC %.4f Pretrain Macro AUROC %.4f Pretrain Macro AUPRC %.4f"
                    % (
                        epoch,
                        step,
                        optimizer.param_groups[0]["lr"],
                        loss.item(),
                        micro_auroc,
                        micro_auprc,
                        macro_auroc,
                        macro_auprc,
                    )
                )

    model.best_model = copy.deepcopy(model.model)


def run_txgnn_pretrained_eval(
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
):
    base.random.seed(seed)
    base.np.random.seed(seed)
    torch.manual_seed(seed)

    project_dir = Path(project_dir)
    split_dir = Path(split_dir) if split_dir is not None else project_dir / "split"
    output_dir = Path(output_dir) if output_dir is not None else split_dir / "txgnn_leakage_free_pretrained_results"
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

    print("Preparing dataframes and graphs for TxGNN with pretraining...")
    frames, split_info = base._prepare_custom_frames(
        project_dir=project_dir,
        split_dir=split_dir,
        output_dir=output_dir,
        valid_ratio=valid_ratio,
        seed=seed,
    )

    train_graph = base.create_dgl_graph(frames["df_train"], frames["df_all"])
    eval_graph = base.create_dgl_graph(frames["df_all"], frames["df_all"])
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
        wandb_run_name = f"TxGNN_Leakage_Free_Pretrained_seed{seed}"

    model = TxGNN(
        data=data_bundle,
        weight_bias_track=use_wandb,
        proj_name=wandb_project,
        exp_name=wandb_run_name or "TxGNN_Leakage_Free_Pretrained_Custom_Split",
        device=device,
    )
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

    print("Starting TxGNN pretraining...")
    _pretrain_with_dgl_compat(
        model=model,
        n_epoch=pretrain_epochs,
        learning_rate=pretrain_lr,
        batch_size=batch_size,
        train_print_per_n=pretrain_print_per_n,
    )

    print("Starting TxGNN finetuning...")
    model.finetune(
        n_epoch=finetune_epochs,
        learning_rate=finetune_lr,
        train_print_per_n=train_print_per_n,
        valid_per_n=valid_per_n,
    )

    model_dir = output_dir / "saved_model_pretrained"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir))
    print(f"Saved pretrained+finetuned TxGNN model to {model_dir}")

    for ntype in eval_graph.ntypes:
        if "inp" in model.G.nodes[ntype].data:
            eval_graph.nodes[ntype].data["inp"] = model.G.nodes[ntype].data["inp"].detach().cpu().clone()
    model.G = eval_graph

    print("Evaluating TxGNN on test diseases...")
    raw_eval = base._run_sparse_safe_disease_eval(
        model=model,
        df_all=frames["df_all"],
        df_train=frames["df_train"],
        df_valid=frames["df_valid"],
        df_test=frames["df_test"],
    )

    summary_df, per_disease_df = base._summarize_official_predictions(
        raw_eval=raw_eval,
        id2name_disease=frames["maps"]["id2name_disease"],
        balanced_repeats=balanced_repeats,
        seed=seed,
    )
    masking_check = base._verify_masking(raw_eval=raw_eval, df_train=frames["df_train"], df_valid=frames["df_valid"])

    summary_path = output_dir / "txgnn_pretrained_summary.csv"
    per_disease_path = output_dir / "txgnn_pretrained_per_disease.csv"
    raw_eval_path = output_dir / "txgnn_pretrained_raw_eval.pkl"
    metadata_path = output_dir / "txgnn_pretrained_metadata.json"
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
    parser = argparse.ArgumentParser(description="Leakage-free TxGNN training with pretraining on a custom disease split.")
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
    args = parser.parse_args()

    summary_df, _, metadata = run_txgnn_pretrained_eval(
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
    )
    print(summary_df.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
