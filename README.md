# Phenotype-Conditioned Drug Repurposing for Undiagnosed Rare Disease Patients

Code and results for our BMI 702 final project: an end-to-end graph–LLM hybrid that ranks drugs from a patient's HPO phenotype set, without requiring a confirmed diagnosis. Built on PrimeKG.

## Repository structure

### `src/data/`
- `primekg_loader.py` — loads PrimeKG and assembles the heterogeneous graph used by all downstream models.
- `disease_split.py` — 80/20 disease-level split, removes test-disease edges from the training subgraph to prevent leakage.
- `text_description_gen.py` — utilities for generating LLM-ready disease/phenotype text descriptions.
- `primKG_preprocess.ipynb`, `train-test.ipynb` — notebook drivers for the preprocessing and split steps.

### `src/baselines/`
- `ppr_baseline.py` — Personalized PageRank seeded from each test disease's phenotype nodes.
- `cascade_baseline.py` — PubCaseFinder → TxGNN cascade with K-sweep.
- `llm_baseline.py` — zero-shot GPT-3.5 / GPT-4o prompted with HPO terms, returns ranked drug list.

### `src/models/`
- `rgcn_encoder.py`, `inductive_encoder.py` — R-GCN encoders over the PrimeKG heterogeneous graph.
- `cross_attention_scorer.py` — drug-conditioned cross-attention scorer.
- `fusion.py`, `feature_fusion_train.py` — feature-level and late-fusion modules combining frozen LLM embeddings with R-GCN node features.
- `train.py` — end-to-end training loop on indication and off-label edges.
- `rgcn_only_pipeline.ipynb`, `feature-level-fusion.ipynb`, `late_fusion_pipeline.ipynb`, `rgcn_LLM_hybrid.ipynb` — notebook pipelines for each model variant; `rgcn_LLM_hybrid.ipynb` is the main reported configuration.

### `src/evaluation/`
- `metrics.py` — MRR and Recall@K computation.
- `significance_tests.py` — paired significance tests across methods.
- `error_analysis.py`, `late_fusion_eval.py` — per-disease stratified analysis and fusion-specific evaluation.
- `LLM_eval.ipynb` — GPT-3.5 / GPT-4o output parsing, drug-name normalization to PrimeKG.
- `all_evaluate_metrics_compare.ipynb` — aggregate metric comparison across all methods on both test splits.
- `fusion_error_analysis_ipynb.ipynb` — stratified analysis (n_phenotypes, n_true_drugs, margin_gap, top10_jaccard), cold-start analysis, attention case studies.

### `src/utils/`
- `negative_sampling.py` — drug-level negative sampling for link-prediction training.
- `debiasing.py` — popularity / degree debiasing utilities.

### `scripts/`
- `generate_descriptions.py` — batch generation of disease and phenotype text descriptions for the LLM encoder.
- `cache_embeddings.py` — caches frozen LLM embeddings to disk so notebooks can reuse them without re-querying.

### `notebooks/`
Standalone exploratory notebooks for fusion variants and embedding-dimension sweeps:
- `feature_fusion_degree_cond.ipynb`, `feature_fusion_autoencoder.ipynb` — degree-conditioned and autoencoder feature-fusion experiments.
- `late_fusion_pipeline.ipynb` — score-level late fusion with per-disease conditioned β gate.
- `llm_embed_dim_256.ipynb` — embedding-dimension ablation.

### `results/`
- `final_all_test_summary.csv` — final MRR / Recall@K results on all 108 test diseases.
- `final_undiagnosed_only_summary.csv` — same metrics on the 78-disease undiagnosable subset.

### `reports/`
- `BMI702_Project_Proposal.pdf` — initial proposal.
- `BMI702_Midterm_report.pdf` — midterm progress report.

## Reproducing results
Run in this order: `src/data/` (preprocessing + split) → `scripts/generate_descriptions.py` and `scripts/cache_embeddings.py` (LLM text + embeddings) → `src/baselines/` → `src/models/` → `src/evaluation/`. Trained checkpoints and intermediate text embeddings are cached in each notebook's working directory.

## Compute
Single NVIDIA A100 (Google Colab Pro). Full R-GCN training runs in ~5 hours; full pipeline including baselines and fusion variants in roughly half a day.
