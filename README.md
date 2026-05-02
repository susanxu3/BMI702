# Phenotype-Conditioned Drug Repurposing for Undiagnosed Rare Disease Patients

Code and results for our BMI 702 final project: an end-to-end graph–LLM hybrid that ranks drugs from a patient's HPO phenotype set, without requiring a confirmed diagnosis. Built on PrimeKG.

## Repository structure

### `data_preparation/`
- `primKG_preprocess.ipynb` — loads PrimeKG, filters to diseases with phenotype and indication edges, builds the heterogeneous graph used by all downstream models.
- `train-test.ipynb` — 80/20 disease-level split, removes test-disease edges from the training subgraph to prevent leakage.

### `baseline/`
- `pagerank.ipynb` — Personalized PageRank seeded from each test disease's phenotype nodes.
- `cascade_no_leakage.ipynb` — PubCaseFinder → TxGNN cascade with K-sweep.
- `llm_baseline_gpt4o.ipynb` — zero-shot GPT-3.5 prompted with HPO terms, returns ranked drug list.

### `model/`
- `rgcn_only_pipeline.ipynb` — R-GCN encoder with drug-conditioned cross-attention scorer, trained end-to-end on indication and off-label edges.
- `feature_level_fusion (1).ipynb` — three feature-level fusion variants (degree-conditioned, autoencoder, residual autoencoder) injecting frozen LLM embeddings as node features.
- `late_fusion_pipeline.ipynb` — score-level late fusion with per-disease conditioned β gate.
- `rgcn_LLM_hybrid.ipynb` — combined R-GCN + LLM hybrid wrapper used for the main reported configuration.

### `evaluation/`
- `LLM_eval.ipynb` — GPT-3.5 / GPT-4o output parsing, drug-name normalization to PrimeKG.
- `all_evaluate_metrics_compare.ipynb` — MRR and Recall@K computation across all methods on both test splits.
- `fusion_error_analysis.ipynb` — stratified analysis (n_phenotypes, n_true_drugs, margin_gap, top10_jaccard), cold-start analysis, attention case studies.

### `results/`
- `final_all_test_summary.csv` — final MRR / Recall@K results on all 108 test diseases.
- `final_undiagnosed_only_summary.csv` — same metrics on the 78-disease undiagnosable subset.

### `reports/`
- `BMI702_Project_Proposal.pdf` — initial proposal.
- `BMI702_Midterm_report.pdf` — midterm progress report.

## Reproducing results
Run notebooks in this order: `data_preparation/` → `baseline/` → `model/` → `evaluation/`. Trained checkpoints and intermediate text embeddings are cached in each notebook's working directory.

## Compute
Single NVIDIA A100 (Google Colab Pro). Full R-GCN training runs in ~5 hours; full pipeline including baselines and fusion variants in roughly half a day.
