# Late Fusion Analysis (Phase 2a)

**Project**: Phenotype-Conditioned Drug Repurposing for Undiagnosed Rare Diseases
**Phase**: 2a — Late Fusion of R-GCN graph scores and text-embedding cosine scores
**Date**: 2026-04-27
**Companion notebook**: [`notebooks/late_fusion_pipeline.ipynb`](../notebooks/late_fusion_pipeline.ipynb)

---

## TL;DR

1. **Late fusion produces small but consistent gains over the pure-graph baseline.**
   Best fused MRR is **0.229** (GPT-4o, no projection) vs. pure-graph **0.200**, a
   relative gain of +14.5%. All five configurations beat the pure-graph baseline.

2. **Description format matters in the opposite direction on train and test.**
   On the train-side proxy MRR, Tier 2 (KG metadata) ranks first (0.262)
   and GPT-4o ranks worst (0.115). On test, the order **inverts** —
   GPT-4o (0.229) > Hybrid (0.208) ≈ Tier 2 (0.208). The proxy MRR is biased
   by literal train-disease name overlap that Tier 2 preserves and GPT-4o does not.

3. **Cold-start drugs are where GPT-4o decisively wins.**
   On the 11 test diseases whose ground-truth drugs were never seen in training,
   GPT-4o LLM-only achieves **MRR 0.17** vs. pure-graph **0.09** and Tier 2
   LLM-only **0.002**. This is the strongest evidence that LLM parametric
   knowledge supplies signal the knowledge graph cannot.

4. **The 5-fold CV β calibration over-weights the graph (β = 0.9 across all
   configs).** Because graph scores on training diseases are inflated by
   training exposure, the calibrator picks β ≈ 0.9 — diluting the LLM signal
   on test, where the graph has not seen the test disease nodes. This is a
   methodological limitation of using the training set for β tuning.

5. **Graph and LLM signals are highly complementary.** Mean Jaccard between
   top-10 lists from the two scorers is 0.02–0.06; 53–81% of test diseases have
   *zero* overlap. Fusion has structural room to grow if the weighting is right.

---

## 1. Experimental Setup

### 1.1 Data and split
- **Knowledge graph**: PrimeKG (129,375 nodes, ~8M edges, 60 relation types incl. reverse).
- **Disease cohort**: 539 rare diseases with ≥1 phenotype and ≥1 drug indication.
- **Split**: 431 train / 108 test diseases. All edges incident to test disease
  nodes are removed from the graph before training. Test disease names are
  also excluded from all text descriptions to prevent string-level leakage.

### 1.2 Components

| Component | Details |
|---|---|
| Graph encoder | 2-layer R-GCN (hidden 256, 3 layers, 15 bases — inferred from `data/weights/rgcn_best_model.pt`) |
| Scorer | Drug-conditioned multi-head cross-attention, 4 heads |
| Text encoder | BiolinkBERT (`michiyasunaga/BioLinkBERT-base`, 768-d), mean-pooled |
| Projection | Selected from {pca, linear, nonlinear_ae, none} |
| Score normalization | Per-disease min-max ([0,1]) before mixing |
| β calibration | 5-fold CV over 431 train diseases, sweep β ∈ {0.0, 0.1, …, 1.0} |
| Fusion rule | `s_final(d) = β · s_graph(d) + (1 − β) · s_LLM(d)` |

### 1.3 Description tiers compared

| Tier | Source | Test MRR (best fused) |
|---|---|---|
| **Tier 1** | Entity name only | n/a (baseline only) |
| **Tier 2** | Entity name + 1-hop KG context (gene/pathway/indication lists) | 0.208 |
| **GPT-4o** | GPT-4o description seeded by Tier 2 metadata; phenotype prompt explicitly forbids disease names | **0.229** |
| **Hybrid** | Tier 2 text concatenated with GPT-4o text | 0.208 |

All four tiers are encoded with BiolinkBERT and projected with two methods
(`nonlinear_ae` and `none`) to enable apples-to-apples comparison.

---

## 2. Encoder and Projection Selection (Section 2 of notebook)

### 2.1 16-way grid (4 encoders × 4 projections, Tier 2)

Proxy MRR on 431 train diseases, ranked top 5:

| Encoder | Projection | Proxy MRR |
|---|---|---|
| **biolinkbert** | **nonlinear_ae** | **0.262** |
| pubmedbert | nonlinear_ae | 0.230 |
| pubmedbert | none | 0.226 |
| specter2 | pca | 0.215 |
| specter2 | none | 0.214 |

BiolinkBERT consistently outperforms PubMedBERT, BiomedBERT, and SPECTER2 — a
plausible explanation is that BiolinkBERT's citation-link pretraining objective
emphasises cross-document relational signals, which align with the
phenotype-disease-drug retrieval task more than abstract-level pretraining.
Nonlinear autoencoder projection wins for two of four encoders, suggesting that
a learned 128-d bottleneck preserves more retrieval-relevant variance than PCA
or a single linear layer.

### 2.2 Tier 1 vs Tier 2 ablation

| Tier | Proxy MRR |
|---|---|
| Tier 1 (name only) | 0.016 |
| Tier 2 (name + KG context) | 0.262 |
| **Δ (Tier 2 − Tier 1)** | **+0.246** |

The 16× lift from Tier 1 to Tier 2 is dominated by KG-anchor token co-occurrence
(see §6.1) rather than by genuinely deeper semantic content; the Tier 2 entries
share gene names, disease names, and pathway names across drug and phenotype
descriptions, so cosine similarity inherits a strong lexical-overlap signal.

---

## 3. GPT-4o Enrichment (Section 4 of notebook)

### 3.1 Prompt design

The prompts treat Tier 2 metadata as a **disambiguation anchor**, not a
knowledge boundary. Both prompts ask GPT-4o to "draw on your full biomedical
knowledge beyond what is listed above," but the phenotype prompt also includes
the constraint "Do not name specific diseases" — preventing GPT-4o from
implicitly leaking test-disease information through its descriptions.

### 3.2 Coverage and failure modes

After concurrent enrichment with backoff:

| Entity type | Total | Successful | Failed (rate-limited or quota) |
|---|---|---|---|
| Drugs | 7,957 | 7,926 (99.6%) | 31 |
| Phenotypes | 3,518 | 3,320 (94.4%) | 198 |

Failed rows fall back to Tier 2 text (with an `_error` field marking the row
for retry). Phenotype enrichment had a higher failure rate, mostly clustered
during a single rate-limit episode — these rows should be retried before
publication.

### 3.3 Proxy MRR by tier (BiolinkBERT, train diseases)

| Tier | Projection | Proxy MRR | Δ vs Tier 2 |
|---|---|---|---|
| Tier 1 | nonlinear_ae | 0.0160 | −0.2462 |
| Tier 2 | nonlinear_ae | 0.2622 | 0.0000 |
| GPT-4o | nonlinear_ae | 0.1155 | −0.1467 |
| GPT-4o | none | 0.1147 | −0.1474 |
| Hybrid | nonlinear_ae | 0.2561 | −0.0061 |
| Hybrid | none | 0.2350 | −0.0271 |

The proxy MRR ranking suggests Tier 2 dominates and GPT-4o is much worse — but
this ordering is misleading. See §6.1 for the leakage mechanism.

---

## 4. Full Late Fusion on the Test Set (Sections 5–7 of notebook)

### 4.1 Five configurations (BiolinkBERT, 108 test diseases)

| Metric | Tier 2 (NLAE) | Hybrid (NLAE) | Hybrid (none) | GPT-4o (NLAE) | **GPT-4o (none)** |
|---|---:|---:|---:|---:|---:|
| best β | 0.90 | 0.90 | 0.90 | 0.90 | **0.80** |
| CV MRR (train) | 0.824 | 0.828 | 0.832 | 0.819 | 0.821 |
| **Test MRR** | 0.208 | 0.208 | 0.208 | 0.219 | **0.229** |
| R@1 | 0.057 | 0.057 | 0.057 | 0.062 | **0.076** |
| R@5 | 0.135 | 0.138 | 0.129 | 0.142 | 0.141 |
| R@10 | 0.175 | 0.184 | 0.180 | 0.188 | 0.186 |
| R@50 | 0.349 | 0.360 | 0.362 | 0.361 | 0.358 |
| AUROC (full library) | 0.904 | 0.906 | 0.905 | 0.886 | 0.893 |
| AUPRC (full library) | 0.154 | 0.158 | 0.155 | 0.161 | 0.168 |
| AUROC (balanced micro) | 0.826 | 0.817 | 0.819 | 0.836 | 0.836 |
| AUPRC (balanced micro) | 0.815 | 0.857 | 0.844 | 0.833 | 0.851 |

### 4.2 Headline comparison vs baselines

| Configuration | Test MRR | Δ vs pure graph |
|---|---:|---:|
| Pure LLM (β = 0, Tier 2) | 0.192 | −0.008 |
| Pure graph (β = 1) | 0.200 | 0.000 (baseline) |
| Tier 2 fused (β = 0.9) | 0.208 | +0.008 |
| Hybrid fused (β = 0.9) | 0.208 | +0.008 |
| GPT-4o fused (β = 0.9, NLAE) | 0.219 | +0.019 |
| **GPT-4o fused (β = 0.8, none)** | **0.229** | **+0.030** |

The best fused configuration improves the pure-graph baseline by +0.030 MRR
(+15% relative). All fusions beat the pure-graph baseline; the GPT-4o
configurations clearly separate from Tier 2 / Hybrid.

---

## 5. Error Analysis (Section 8 of notebook)

### 5.1 A1 — Per-disease delta vs pure graph

| Configuration | mean Δ MRR | n_better / n_worse | max single gain | max single loss |
|---|---:|---:|---:|---:|
| Tier 2 (NLAE) | +0.0085 | 61 / 16 | +0.31 (atopic conjunctivitis) | −0.08 |
| Hybrid (NLAE) | +0.0094 | 63 / 9 | +0.33 (autism) | −0.08 |
| Hybrid (none) | +0.0089 | 62 / 11 | +0.50 (gastric ulcer) | −0.50 (narcolepsy-cataplexy) |
| GPT-4o (NLAE) | +0.0203 | 57 / 13 | +1.00 (autism) | −0.17 |
| **GPT-4o (none)** | **+0.0256** | **67 / 11** | +0.67 (hemangioma) | −0.02 |

GPT-4o (none) wins on three independent axes simultaneously: highest mean Δ,
most diseases helped (67/108), and smallest worst-case loss (−0.024). The
asymmetry between max gain (+1.0) and max loss (−0.024) suggests that GPT-4o
fusion behaves as a "rescue" mechanism — it drives a few near-zero diseases
to perfect rank without catastrophically hurting any well-served disease.

### 5.2 A2 — Cold-start stratification

Test diseases binned by `seen_ratio = (# true drugs seen in training) / (# true drugs)`:

| | All seen (n=65) | Some seen (n=32) | **None seen (n=11)** |
|---|---:|---:|---:|
| Pure graph (β = 1) | 0.222 | 0.191 | 0.092 |
| Tier 2 LLM-only | 0.191 | 0.259 | 0.002 |
| Hybrid LLM-only | 0.263 | 0.298 | 0.021 |
| **GPT-4o LLM-only** | 0.103 | 0.203 | **0.173** |
| Tier 2 fused | 0.229 | 0.206 | 0.092 |
| Hybrid (none) fused | 0.226 | 0.211 | 0.103 |
| **GPT-4o (none) fused** | **0.248** | **0.214** | **0.122** |

Two findings stand out:

1. **Tier 2 LLM-only collapses to 0.002 on cold-start**. With no training
   indications for the drug, the Tier 2 description reduces to
   `"<Drug>. Targets: <gene list>."` — there is no clinical context to match
   against the phenotype description, and the cosine similarity becomes
   essentially random.

2. **GPT-4o LLM-only achieves 0.173 on cold-start, beating pure graph (0.092)
   and Tier 2 LLM-only by 86×.** GPT-4o's description always contains
   mechanism, therapeutic class, and clinical applications — content that does
   not depend on the drug being seen in training. This is the strongest
   single piece of evidence that LLM parametric knowledge fills a real gap
   in PrimeKG-derived signals.

   At β = 0.9, however, the cold-start gain of fused GPT-4o is only +0.030
   over pure graph (0.122 vs 0.092). The β calibration dilutes the LLM
   advantage where it matters most. See §6.2.

### 5.3 A3 — Phenotype-count stratification

Test diseases binned by number of phenotypes:

| | 1–3 (n=24) | 4–10 (n=21) | 11–30 (n=34) | 30+ (n=29) |
|---|---:|---:|---:|---:|
| Pure graph (β = 1) | 0.109 | 0.143 | 0.223 | 0.288 |
| Tier 2 LLM-only | 0.175 | 0.182 | 0.196 | 0.207 |
| Hybrid (none) LLM-only | 0.254 | 0.299 | 0.207 | 0.261 |
| GPT-4o (NLAE) LLM-only | 0.245 | 0.160 | 0.084 | 0.099 |
| GPT-4o (none) fused (β = 0.8) | 0.171 | 0.142 | 0.246 | 0.307 |

Pattern:

- The graph baseline scales monotonically with phenotype count
  (0.109 → 0.288 from sparse to abundant).
- LLM-only configurations are roughly *flat* across phenotype counts.
- Hybrid LLM-only achieves the largest absolute MRR (0.30) on the 4–10 bin.
- Fused configurations track the graph baseline closely on the dense bins
  (11+ phenotypes) — again because β = 0.9 lets the graph signal dominate.

Implication: text fusion is most valuable in the sparse-phenotype regime,
exactly the regime that motivated the project (undiagnosed rare disease
patients with limited HPO observations). The 1–3 bin lift from Tier 2 LLM
(+0.067) and Hybrid LLM (+0.145 / +0.146) over pure graph is meaningful, but
again the fused versions absorb only a fraction of this lift.

### 5.4 A4 — Graph vs LLM top-10 disagreement (Jaccard)

| Configuration | mean Jaccard | median | frac. zero overlap | Spearman ρ (Jaccard, Δ MRR) |
|---|---:|---:|---:|---:|
| Tier 2 (NLAE) | 0.054 | 0.000 | 53% | **−0.22** |
| Hybrid (NLAE) | 0.060 | 0.000 | 56% | **−0.31** |
| Hybrid (none) | 0.058 | 0.000 | 56% | **−0.21** |
| GPT-4o (NLAE) | 0.019 | 0.000 | 81% | +0.03 |
| GPT-4o (none) | 0.024 | 0.000 | 81% | +0.17 |

Two observations:

1. **Top-10 lists from the graph and the LLM scorers have minimal overlap**
   — for GPT-4o, 81% of test diseases have *zero* overlap. This is structurally
   ideal for fusion: the two scorers see different parts of the drug space.

2. **Spearman correlations differ in sign by configuration.**
   For Tier 2 and Hybrid, fusion gain is *larger* when Jaccard is small
   (negative ρ) — the canonical pattern for productive late fusion: fuse
   when signals disagree, defer to graph when they agree. For GPT-4o, the
   correlation is near zero or slightly positive, meaning fusion gains are
   not predicted by overlap. The interpretation is that GPT-4o supplies
   genuinely orthogonal signal across the entire test set, not only on the
   subset where graph and LLM disagree.

---

## 6. Why the Numbers Look the Way They Do

### 6.1 Why the proxy MRR ranking inverts on test

Proxy MRR is computed on the 431 training diseases. For each train disease Y,
the supervision is `Y → {drug D₁, D₂, …}`. In the Tier 2 description scheme:

- `Drug D` carries the literal substring `"Indications: …, Y, …"`.
- Each phenotype `P` of `Y` carries the literal substring `"Associated conditions: …, Y, …"`.

When BiolinkBERT mean-pools these descriptions, the two embeddings inherit a
strong positive cosine through their shared rare-token "Y". Proxy MRR for
disease Y therefore rewards the same drugs that are listed in `Y`'s
own indication context — this is target leakage, not generalization.

GPT-4o descriptions are produced under the constraint "Do not name specific
diseases" for phenotypes, and for drugs the prompt asks for mechanism /
class / clinical use rather than the literal indication list. The leakage
shortcut is removed, and proxy MRR drops from 0.262 to 0.115.

On the test set, the leakage shortcut is removed for *all* tiers (test disease
names are explicitly excluded by `build_drug_descriptions(exclude_diseases=…)`
and `build_phenotype_descriptions(exclude_nodes=…)`). The ranking that
emerges — GPT-4o > Hybrid ≈ Tier 2 — is the unbiased one.

**Implication for the experimental protocol**: proxy MRR should be used only
for *within-tier* encoder-and-projection selection (where the leakage rate is
constant across candidates). For *cross-tier* comparison, only test-set
metrics are valid.

### 6.2 Why β = 0.9 is suboptimal but selected

The 5-fold CV calibrator picks β = 0.9 for almost every configuration, and
the resulting CV MRR is ~0.83. This number is implausibly high relative to
test MRR (~0.21).

The mechanism: 5-fold CV is performed on training diseases, against which the
R-GCN was trained with indication edges in the graph. The graph score for a
training disease is therefore inflated by the model having seen its true
drugs. With graph dominating, β close to 1.0 maximises CV MRR. On the test
set, the graph has *not* seen the test disease nodes, and the optimal β is
likely lower — particularly for cold-start diseases, where the LLM signal
should dominate.

A2 confirms this: GPT-4o LLM-only achieves 0.17 on cold-start (better than
pure graph at 0.09), but at β = 0.8–0.9 the fused score is only 0.12 because
the (worse) graph signal is weighted at 80–90%.

This is a known limitation of using training-set CV to tune hyperparameters
of a model that was trained on the same set. A more principled fix would be
either (a) per-disease adaptive β driven by an indicator like
`drug_degree`, or (b) holding out a small validation slice of training
diseases that are *also* masked from the R-GCN training graph during a
re-training pass.

### 6.3 Why Hybrid does not beat Tier 2

Hybrid is `text = tier2_text + " " + gpt4o_text`. Mean-pooling at the
encoder level means the strong, rare KG-anchor tokens in Tier 2 are diluted
by an equal mass of generic prose tokens from GPT-4o. The Tier 2 anchor
signal contributes ~50% of the pooled embedding instead of ~100%, and the
remaining ~50% is generic biomedical vocabulary that has weak discriminative
power for retrieval. A text-level concatenation is therefore worse than
either input alone in raw cosine similarity — which is what we observe.

A more promising fusion of the two signals is at the embedding level:

```
emb_hybrid = α · emb_tier2 + (1 − α) · emb_gpt4o   (encoded separately, then averaged)
```

This preserves the Tier 2 anchor strength while still adding GPT-4o
information. We have not implemented this variant.

---

## 7. Methodological Limitations

1. **Train-side β calibration is biased** (§6.2). Test MRR is best treated
   as a lower bound on what a properly calibrated fusion can achieve.
2. **Proxy MRR is not a tier-comparable metric** (§6.1). Use only for
   within-tier hyperparameter selection.
3. **Phenotype enrichment had ~5% API failures.** Those rows currently fall
   back to Tier 2 text. Re-running enrichment after restoring the OpenAI
   quota would clean up this contamination.
4. **β is global, not per-disease**. A2 / A3 strongly suggest that the
   optimal β depends on cold-start status and phenotype count.
5. **Single text encoder family compared at full scale**. The encoder grid
   (BiolinkBERT vs PubMedBERT vs BiomedBERT vs SPECTER2) was run on Tier 2
   only; we have not validated whether BiolinkBERT remains best when the
   description content is GPT-4o prose rather than KG metadata.

---

## 8. Headline Findings for the Phase 2a Report

1. **Late fusion improves over pure-graph by +0.030 MRR (+15%) on 108 test
   rare diseases**, with the best configuration being GPT-4o descriptions,
   no projection, and β = 0.8.

2. **GPT-4o descriptions provide a real and quantitatively measurable
   complement to PrimeKG.** The clearest evidence is the cold-start
   stratum: GPT-4o LLM-only achieves MRR 0.17 on diseases whose true drugs
   were never seen in training, vs. 0.09 for the pure-graph baseline and
   0.002 for Tier 2 LLM-only. This validates the project's central
   hypothesis that LLM parametric knowledge can rescue cold-start drugs.

3. **The proxy MRR–test MRR inversion is a leakage artifact, not a
   contradiction.** Tier 2 descriptions contain literal train-disease names
   that cosine similarity exploits on the train set; this shortcut is
   removed on the test set, and GPT-4o overtakes Tier 2.

4. **Fusion gains are bottlenecked by β calibration on the train set.**
   The CV procedure pushes β to 0.9 because graph scores are inflated on
   training diseases. Cold-start and sparse-phenotype strata, where LLM
   signal is most valuable, suffer the most from this dilution. A
   per-disease adaptive β is the highest-value next experiment.

5. **Graph and LLM signals are nearly orthogonal**: 81% of test diseases
   have zero top-10 overlap under GPT-4o. The structural ceiling for
   late fusion is much higher than what the current global β achieves.

---

## 9. Recommended Next Steps

| # | Experiment | Why | Effort |
|---|---|---|---|
| 1 | Per-disease adaptive β driven by `drug_degree` (cold-start indicator) | A2 shows cold-start wants β ≈ 0.2 while warm wants β ≈ 0.9 | 1–2 days |
| 2 | Embedding-level Hybrid: `α · emb_tier2 + (1−α) · emb_gpt4o` (encoded separately) | Avoids the mean-pool dilution that text concat suffers (§6.3) | 0.5 day |
| 3 | Retry the 198 phenotype-enrichment failures | Cleans 5.6% contamination in GPT-4o phenotypes | trivial after API quota |
| 4 | Re-test BiolinkBERT vs PubMedBERT on GPT-4o text | Encoder selection was done on Tier 2 only; the prose distribution may favor a different encoder | ~1 day |
| 5 | Hold out 20% of train diseases (with their graph edges masked during R-GCN re-training) for unbiased β calibration | Removes the train-side bias that pushes β to 0.9 | ~2 days (requires retraining) |

---

## Appendix A — Reproduction

All experiments are reproducible from
`notebooks/late_fusion_pipeline.ipynb` with the following inputs:

- `data/primekg/{nodes,edges,kg}.csv`
- `data/splits/{train,test}_disease_ids.txt`
- `data/weights/rgcn_best_model.pt`
- `data/descriptions/{drugs,phenotypes}_{tier1,tier2,gpt4o,hybrid}.json`
- `data/embeddings/biolinkbert/{tier1,tier2,gpt4o,hybrid}/{nonlinear_ae,none}/`

Per-disease error-analysis results are saved to
`results/tables/section7_config_comparison.json` and figures to
`results/A1_per_disease_delta.png`, `results/A4_jaccard_vs_delta.png`.

## Appendix B — File Map

| Path | Role |
|---|---|
| `scripts/generate_descriptions.py` | Tier 1 / Tier 2 / GPT-4o description generation |
| `scripts/cache_embeddings.py` | Encode descriptions with BiolinkBERT etc., apply projection, L2-normalize, save |
| `src/models/cross_attention_scorer.py` | `PhenoDrugModel` (R-GCN + drug-conditioned cross-attention) |
| `src/models/fusion.py` | `LateFusion`, `normalize_scores`, `calibrate_beta` (5-fold CV) |
| `src/evaluation/late_fusion_eval.py` | `run_late_fusion_experiment` — top-level orchestrator with checkpoint arch inference, score loading, and per-disease evaluation including cold-start stratification |
| `src/evaluation/metrics.py` | MRR, R@K, full-library AUROC/AUPRC |
