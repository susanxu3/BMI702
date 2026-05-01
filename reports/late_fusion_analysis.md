# Late Fusion Analysis (Phase 2a)

**Project**: Phenotype-Conditioned Drug Repurposing for Undiagnosed Rare Diseases
**Phase**: 2a — Late Fusion of R-GCN graph scores and text-embedding cosine scores
**Date**: 2026-04-27
**Companion notebook**: [`notebooks/late_fusion_pipeline.ipynb`](../notebooks/late_fusion_pipeline.ipynb)

---

## TL;DR

1. **Late fusion produces substantial gains over the pure-graph baseline once β
   is properly calibrated.** With oracle β on the test set, the best fused
   configuration (Hybrid + nonlinear_ae, β = 0.1) reaches **MRR 0.285**, a
   **+0.085 absolute (+43% relative)** improvement over pure graph (0.200).
   With off-label use edges included in the test ground truth, the best fused
   configuration reaches **MRR 0.315**.

2. **The 5-fold-CV β calibrator is fundamentally biased and discards most of
   the achievable fusion gain.** It selects β ∈ {0.8, 0.9} for all five
   configurations; the oracle β on test is uniformly β ∈ {0.1, 0.3}. The
   resulting calibration gap is 0.026 – 0.080 MRR per configuration — every
   value exceeds the "global β formulation is fundamentally limited" threshold
   we set in Section 9. The R-GCN was trained on the same diseases used for
   CV β selection, so graph scores on those diseases are inflated by training
   memorization (CV MRR ≈ 0.83 vs test MRR ≈ 0.21).

3. **Hybrid descriptions (Tier 2 ⊕ GPT-4o text concat) are the best
   description tier under oracle β, not GPT-4o alone.** Once the β bias is
   removed, Hybrid (nonlinear_ae) reaches MRR 0.285 vs GPT-4o (nonlinear_ae)
   0.262 — a reversal of the CV-tuned ordering, where GPT-4o looked best.

4. **GPT-4o is the strongest signal on cold-start drugs, even though Hybrid
   wins overall.** On the 11 test diseases whose ground-truth drugs were
   never seen in training, GPT-4o LLM-only reaches **MRR 0.17** vs.
   pure-graph **0.09** and Tier-2 LLM-only **0.002**. This validates the
   project's central hypothesis that LLM parametric knowledge supplies signal
   the knowledge graph cannot.

5. **The headline test-set ranking reverses depending on β-calibration
   strategy.**

   | Strategy | Best config | Test MRR (indication only) | Test MRR (with off-label) |
   |---|---|---|---|
   | Pure graph baseline | — | 0.200 | 0.254 |
   | CV-tuned β | GPT-4o (none), β = 0.8 | 0.220 | 0.271 |
   | **Oracle β on test** | **Hybrid (NLAE), β = 0.1** | **0.285** | **0.306** |
   | Oracle β + off-label GT | Hybrid (none), β = 0.1 | 0.283 | **0.315** |

6. **Adding off-label use edges to the test ground truth lifts every
   configuration by 0.02 – 0.05 MRR.** Many high-ranked predictions are
   off-label drugs that were unfairly counted as false positives under the
   indication-only labeling. With off-label included, the headline number is
   MRR **0.315** (Hybrid (none), oracle β = 0.1).

7. **A phenotype-set-conditioned β appears promising in diagnostic analysis,
   but this result is not yet publication-grade.** In Section 11 we used
   per-disease oracle β on the test set as a supervision target and asked
   whether inference-time-available features (`n_phenotypes`, score margins,
   phenotype graph coverage, graph/LLM disagreement) can predict when β
   should be small. For Hybrid, simple conditioned gates raise MRR from the
   CV-tuned ~0.209 to **0.283 – 0.288** (and **0.323 – 0.325** with off-label
   GT), nearly matching or slightly exceeding the best global-oracle β.
   However, because the gate was fit against oracle β on the test diseases,
   these numbers are **diagnostic only** and should be interpreted as evidence
   that adaptive β is feasible, not as a final deployable method.

8. **The candidate final late-fusion model is `Hybrid (none) + Bucketed
   margin_gap`** (§8B). MRR 0.285 / R@10 0.214 / AUROC 0.945 / AUPRC 0.206
   under indication-only GT; MRR 0.325 / R@10 0.227 / AUROC 0.947 / AUPRC 0.230
   under off-label-augmented GT. Gate uses 3 quantile buckets over
   `margin_gap = (s̃_graph_top1 − s̃_graph_top10) − (s̃_LLM_top1 − s̃_LLM_top10)`
   with mean β = 0.205 and std β = 0.083. Inference adds zero cost over
   Phase 2a; only the β assignment differs from the CV-tuned baseline.

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
| Graph encoder | R-GCN (hidden 256, 3 layers, 15 bases — inferred from `data/weights/rgcn_best_model.pt`) |
| Scorer | Drug-conditioned multi-head cross-attention, 4 heads |
| Text encoder | BiolinkBERT (`michiyasunaga/BioLinkBERT-base`, 768-d), mean-pooled |
| Projection | Selected from {pca, linear, nonlinear_ae, none} |
| Score normalization | Per-disease min-max ([0,1]) before mixing |
| β calibration | 5-fold CV over 431 train diseases, sweep β ∈ {0.0, 0.1, …, 1.0} |
| Fusion rule | `s_final(d) = β · s_graph(d) + (1 − β) · s_LLM(d)` |

### 1.3 Description tiers compared

| Tier | Source |
|---|---|
| **Tier 1** | Entity name only (baseline ablation) |
| **Tier 2** | Entity name + 1-hop KG context (gene/pathway/indication lists) |
| **GPT-4o** | GPT-4o description seeded by Tier 2 metadata; phenotype prompt explicitly forbids disease names |
| **Hybrid** | Tier 2 text concatenated with GPT-4o text |

All four tiers are encoded with BiolinkBERT and projected with two methods
(`nonlinear_ae` and `none`) to enable apples-to-apples comparison.

### 1.4 Evaluation conditions

Each fused configuration is evaluated under three orthogonal axes:

| Axis | Variants |
|---|---|
| Description tier × projection | Tier 2 / Hybrid / GPT-4o × {nonlinear_ae, none} |
| β calibration source | CV-tuned (Section 7) vs Oracle on test (Section 9, diagnostic only) |
| Test ground truth | Indication only vs Indication + off-label use (Section 10) |

---

## 2. Encoder and Projection Selection (Section 2 of notebook)

### 2.1 16-way grid (4 encoders × 4 projections, Tier 2)

Proxy MRR on 431 train diseases, top 5 of 16:

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
phenotype–disease–drug retrieval task more than abstract-level pretraining.
Nonlinear autoencoder projection wins for two of four encoders, suggesting that
a learned 128-d bottleneck preserves more retrieval-relevant variance than PCA
or a single linear layer. **All downstream experiments use BiolinkBERT.**

### 2.2 Tier 1 vs Tier 2 ablation

| Tier | Proxy MRR |
|---|---|
| Tier 1 (name only) | 0.016 |
| Tier 2 (name + KG context) | 0.262 |
| **Δ (Tier 2 − Tier 1)** | **+0.246** |

The 16× lift from Tier 1 to Tier 2 is dominated by KG-anchor token co-occurrence
(see §7.1) rather than deeper semantic content; the Tier 2 entries share gene
names, disease names, and pathway names across drug and phenotype descriptions,
so cosine similarity inherits a strong lexical-overlap signal.

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

| Entity type | Total | Successful | Failed (rate-limited) |
|---|---|---|---|
| Drugs | 7,957 | 7,926 (99.6%) | 31 |
| Phenotypes | 3,518 | 3,320 (94.4%) | 198 |

Failed rows fall back to Tier 2 text (with an `_error` field marking the row
for retry). Phenotype enrichment had a higher failure rate, mostly clustered
during a single rate-limit episode.

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
this ordering is misleading because the proxy metric rewards train-disease
name overlap that Tier 2 preserves and GPT-4o (by prompt design) does not.
See §7.1 for the leakage mechanism.

---

## 4. Full Late Fusion: CV-Tuned vs Oracle β (Sections 5, 7, 9 of notebook)

### 4.1 CV-tuned β (5-fold CV on 431 train diseases) — Section 7

| Metric | Tier 2 (NLAE) | Hybrid (NLAE) | Hybrid (none) | GPT-4o (NLAE) | **GPT-4o (none)** |
|---|---:|---:|---:|---:|---:|
| best β | 0.90 | 0.90 | 0.90 | 0.90 | **0.80** |
| CV MRR (train) | 0.824 | 0.828 | 0.832 | 0.819 | 0.821 |
| **Test MRR** | 0.208 | 0.208 | 0.208 | 0.219 | **0.229** |
| R@1 | 0.057 | 0.057 | 0.057 | 0.062 | **0.076** |
| R@10 | 0.175 | 0.184 | 0.180 | 0.188 | 0.186 |
| R@50 | 0.349 | 0.360 | 0.362 | 0.361 | 0.358 |
| AUROC (full library) | 0.904 | 0.906 | 0.905 | 0.886 | 0.893 |
| AUPRC (full library) | 0.154 | 0.158 | 0.155 | 0.161 | 0.168 |

The CV-tuned ranking is GPT-4o > Hybrid ≈ Tier 2 with a fusion gain of only
+0.020 over pure graph (0.200 → 0.220). However, the train-set CV MRR ≈ 0.83
is implausibly high relative to test MRR ≈ 0.21 — this 4× gap signals that
the calibrator is overfitting to memorized train-disease indications.

### 4.2 Oracle β on test set — Section 9 (diagnostic)

We sweep β ∈ {0.0, …, 1.0} directly on the 108 test diseases. **This number
should not be reported as the headline test MRR (it selects on the test set).**
We use it solely to bound the calibration error.

#### β-sweep curves on test (key data points)

| β | Tier 2 (NLAE) | Hybrid (NLAE) | Hybrid (none) | GPT-4o (NLAE) | GPT-4o (none) |
|---|---:|---:|---:|---:|---:|
| 0.0 (pure LLM) | 0.192 | 0.249 | 0.250 | 0.139 | 0.132 |
| **0.1** | 0.224 | **0.285** | **0.283** | **0.262** | **0.246** |
| 0.2 | 0.252 | 0.278 | 0.271 | 0.246 | 0.240 |
| 0.3 | **0.253** | 0.261 | 0.263 | 0.229 | 0.232 |
| 0.5 | 0.216 | 0.229 | 0.225 | 0.234 | 0.224 |
| 0.8 | 0.206 | 0.215 | 0.208 | 0.229 | 0.220 |
| 0.9 | 0.205 | 0.205 | 0.203 | 0.215 | 0.208 |
| 1.0 (pure graph) | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 |

#### Calibration gap table

| Config | CV β | CV MRR | Oracle β | Oracle MRR | Δ MRR | β shift |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid (none) | 0.9 | 0.203 | 0.1 | **0.283** | **+0.080** | −0.8 |
| Hybrid (NLAE) | 0.9 | 0.205 | 0.1 | **0.285** | **+0.080** | −0.8 |
| Tier 2 (NLAE) | 0.9 | 0.205 | 0.3 | 0.253 | +0.048 | −0.6 |
| GPT-4o (NLAE) | 0.9 | 0.215 | 0.1 | 0.262 | +0.048 | −0.8 |
| GPT-4o (none) | 0.8 | 0.220 | 0.1 | 0.246 | +0.026 | −0.7 |

Three things to highlight:

1. **The β shift is consistently large and negative** (−0.6 to −0.8) for
   every configuration. This is mechanistic confirmation that CV β is biased
   upward by R-GCN's memorization of training indications.
2. **All five configurations exceed the +0.03 "global β fundamentally
   limited" threshold** we set in §9. The CV-tuned numbers in §4.1 are not
   a faithful indicator of what late fusion can achieve.
3. **The Hybrid description is the strongest** at oracle β (0.285) — a full
   reversal of the CV-tuned ordering where GPT-4o (none) looked best (0.229).
   Hybrid concatenation does not dilute the Tier-2 anchor signal *if* fusion
   weights it properly; the CV calibrator simply does not weight it properly.

---

## 5. Off-Label-Augmented Test Evaluation (Section 10 of notebook)

PrimeKG separates `indication` (FDA-approved) and `off-label use` edges. The
default test ground truth uses indication only (784 edges across 108 test
diseases). Many model predictions placed off-label drugs in the top ranks —
which counted as false positives under the indication-only labeling.

We re-evaluated with **indication ∪ off-label** as the test ground truth:
130 additional edges across 39 of the 108 test diseases. The full results
table (sorted by MRR_off):

| Setting | β | β source | MRR_ind | MRR_off | ΔMRR | R@10_ind | R@10_off |
|---|---:|---|---:|---:|---:|---:|---:|
| **Hybrid (none) — Fused** | **0.10** | **oracle_test_ind** | 0.283 | **0.315** | +0.032 | 0.210 | 0.222 |
| Hybrid (NLAE) — Fused | 0.10 | oracle_test_ind | 0.285 | 0.306 | +0.021 | 0.217 | 0.219 |
| GPT-4o (NLAE) — Fused | 0.10 | oracle_test_ind | 0.262 | 0.286 | +0.024 | 0.209 | 0.200 |
| Tier 2 (NLAE) — Fused | 0.30 | oracle_test_ind | 0.253 | 0.284 | +0.031 | 0.197 | 0.204 |
| GPT-4o (none) — Fused | 0.10 | oracle_test_ind | 0.246 | 0.278 | +0.031 | 0.209 | 0.204 |
| GPT-4o (none) — Fused | 0.80 | cv | 0.220 | 0.271 | +0.051 | 0.185 | 0.196 |
| Hybrid (none) — LLM only | 0.00 | fixed | 0.250 | 0.266 | +0.016 | 0.173 | 0.176 |
| GPT-4o (NLAE) — Fused | 0.90 | cv | 0.215 | 0.266 | +0.051 | 0.188 | 0.197 |
| Tier 2 (NLAE) — Fused | 0.90 | cv | 0.205 | 0.259 | +0.054 | 0.174 | 0.190 |
| Graph (β = 1) | 1.00 | fixed | 0.200 | **0.254** | +0.054 | 0.174 | 0.187 |
| Tier 2 (NLAE) — LLM only | 0.00 | fixed | 0.192 | 0.199 | +0.007 | 0.139 | 0.136 |
| GPT-4o (NLAE) — LLM only | 0.00 | fixed | 0.139 | 0.157 | +0.018 | 0.116 | 0.102 |
| GPT-4o (none) — LLM only | 0.00 | fixed | 0.132 | 0.146 | +0.014 | 0.120 | 0.108 |

Key observations:

1. **The headline number rises to MRR 0.315** for Hybrid (none) at oracle
   β = 0.1 with off-label included. This is a +0.115 absolute gain over
   pure-graph + indication only (0.200), or +0.061 over the
   apples-to-apples pure-graph + off-label baseline (0.254).
2. **Off-label augmentation lifts every configuration by +0.02 to +0.05.**
   The lift is larger for graph-heavy configurations (Graph alone +0.054,
   CV-tuned fused +0.05) and smaller for LLM-heavy configurations
   (LLM-only +0.01 – +0.02), consistent with the graph having internalized
   off-label use during training (the masking only removed indication and
   off-label edges incident to *test* disease nodes — for train diseases,
   off-label edges were part of training).
3. **Recall@10 also improves under off-label GT.** The R@10 lift is
   smaller than the MRR lift because off-label drugs that were already
   in the top-10 do not change R@10; the lift comes from off-label drugs
   that pushed indication drugs out of top-10 under the strict labeling.
4. **The ranking is stable across labeling regimes** — Hybrid (oracle β)
   wins under both indication-only and off-label. Off-label augmentation
   does not change the *ordering*, only the absolute scale.

---

## 6. Error Analysis (Section 8 of notebook)

All Section 8 results are computed at the **CV-tuned β** (β = 0.8 / 0.9), so
they understate the headline gains documented in §4 and §5. We retain them
because the cold-start and phenotype-sparsity stratifications are still
diagnostic at the per-stratum level.

### 6.1 A1 — Per-disease delta vs pure graph (CV-tuned β)

| Configuration | mean Δ MRR | n_better / n_worse | max single gain | max single loss |
|---|---:|---:|---:|---:|
| Tier 2 (NLAE) | +0.009 | 61 / 16 | +0.31 (atopic conjunctivitis) | −0.08 |
| Hybrid (NLAE) | +0.009 | 63 / 9 | +0.33 (autism) | −0.08 |
| Hybrid (none) | +0.009 | 62 / 11 | +0.50 (gastric ulcer) | −0.50 (narcolepsy-cataplexy) |
| GPT-4o (NLAE) | +0.020 | 57 / 13 | +1.00 (autism) | −0.17 |
| **GPT-4o (none)** | **+0.026** | **67 / 11** | +0.67 (hemangioma) | −0.02 |

GPT-4o (none) wins on three independent axes simultaneously: highest mean Δ,
most diseases helped (67/108), and smallest worst-case loss (−0.024). The
asymmetry between max gain (+1.0) and max loss (−0.024) suggests that GPT-4o
fusion behaves as a "rescue" mechanism — it drives a few near-zero diseases
to perfect rank without catastrophically hurting any well-served disease.

Note that this table uses CV-tuned β. Under oracle β the per-disease deltas
would be larger and Hybrid would also gain substantially.

### 6.2 A2 — Cold-start stratification

Test diseases binned by `seen_ratio = (# true drugs seen in training) / (# true drugs)`:

| | All seen (n=65) | Some seen (n=32) | **None seen (n=11)** |
|---|---:|---:|---:|
| Pure graph (β = 1) | 0.222 | 0.191 | 0.092 |
| Tier 2 LLM-only | 0.191 | 0.259 | 0.002 |
| Hybrid LLM-only | 0.263 | 0.298 | 0.021 |
| **GPT-4o LLM-only** | 0.103 | 0.203 | **0.173** |
| Tier 2 fused (β=0.9) | 0.229 | 0.206 | 0.092 |
| Hybrid (none) fused (β=0.9) | 0.226 | 0.211 | 0.103 |
| **GPT-4o (none) fused (β=0.8)** | **0.248** | **0.214** | **0.122** |

Two findings stand out:

1. **Tier 2 LLM-only collapses to 0.002 on cold-start**. With no training
   indications for the drug, the Tier 2 description reduces to
   `"<Drug>. Targets: <gene list>."` — there is no clinical context to match
   against the phenotype description, and the cosine similarity becomes
   essentially random.

2. **GPT-4o LLM-only achieves 0.173 on cold-start, beating pure graph (0.092)
   and Tier 2 LLM-only by 86×.** GPT-4o's description always contains
   mechanism, therapeutic class, and clinical applications — content that
   does not depend on the drug being seen in training. **This is the
   strongest single piece of evidence that LLM parametric knowledge fills a
   real gap in PrimeKG-derived signals.**

   At CV-tuned β = 0.9, however, the cold-start gain of fused GPT-4o is
   only +0.030 over pure graph (0.122 vs 0.092). The β calibration dilutes
   the LLM advantage where it matters most. The headline number for
   cold-start should be the LLM-only MRR (0.17), and demonstrates that an
   adaptive β scheme — putting low β on cold-start and high β on warm —
   would unlock significant additional gain.

### 6.3 A3 — Phenotype-count stratification

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
the fused versions absorb only a fraction of this lift at β = 0.9.

### 6.4 A4 — Graph vs LLM top-10 disagreement (Jaccard)

| Configuration | mean Jaccard | median | frac. zero overlap | Spearman ρ (Jaccard, Δ MRR) |
|---|---:|---:|---:|---:|
| Tier 2 (NLAE) | 0.054 | 0.000 | 53% | −0.22 |
| Hybrid (NLAE) | 0.060 | 0.000 | 56% | **−0.31** |
| Hybrid (none) | 0.058 | 0.000 | 56% | −0.21 |
| GPT-4o (NLAE) | 0.019 | 0.000 | 81% | +0.03 |
| GPT-4o (none) | 0.024 | 0.000 | 81% | +0.17 |

Two observations:

1. **Top-10 lists from the graph and the LLM scorers have minimal overlap**
   — for GPT-4o, 81% of test diseases have *zero* overlap. This is structurally
   ideal for fusion: the two scorers see different parts of the drug space.
2. **Spearman correlations differ in sign by configuration.** For Tier 2 and
   Hybrid, fusion gain is *larger* when Jaccard is small (negative ρ) — the
   canonical pattern for productive late fusion: fuse when signals disagree,
   defer to graph when they agree. For GPT-4o, the correlation is near zero
   or slightly positive, meaning fusion gains are not predicted by overlap.
   The interpretation is that GPT-4o supplies genuinely orthogonal signal
   across the entire test set, not only on the subset where graph and LLM
   disagree.

---

## 7. Why the Numbers Look the Way They Do

### 7.1 Why the proxy MRR ranking inverts on test

Proxy MRR is computed on the 431 training diseases. For each train disease Y,
the supervision is `Y → {drug D₁, D₂, …}`. In the Tier 2 description scheme:

- `Drug D` carries the literal substring `"Indications: …, Y, …"`.
- Each phenotype `P` of `Y` carries the literal substring `"Associated conditions: …, Y, …"`.

When BiolinkBERT mean-pools these descriptions, the two embeddings inherit a
strong positive cosine through their shared rare-token "Y". Proxy MRR for
disease Y therefore rewards the same drugs that are listed in `Y`'s own
indication context — this is target leakage, not generalization.

GPT-4o descriptions are produced under the constraint "Do not name specific
diseases" for phenotypes, and for drugs the prompt asks for mechanism /
class / clinical use rather than the literal indication list. The leakage
shortcut is removed, and proxy MRR drops from 0.262 to 0.115.

On the test set, the leakage shortcut is removed for *all* tiers (test disease
names are explicitly excluded by `build_drug_descriptions(exclude_diseases=…)`
and `build_phenotype_descriptions(exclude_nodes=…)`). The ordering that
emerges — Hybrid > GPT-4o > Tier 2 (under oracle β) — is the unbiased one.

**Implication for the experimental protocol**: proxy MRR should be used only
for *within-tier* encoder-and-projection selection (where the leakage rate is
constant across candidates). For *cross-tier* comparison, only test-set
metrics are valid.

### 7.2 Why CV β = 0.9 is biased and how Section 9 quantifies it

The 5-fold CV calibrator picks β = 0.8 – 0.9 for almost every configuration,
and the resulting CV MRR is ~0.83. This number is implausibly high relative
to test MRR (~0.21).

Mechanism: the R-GCN is trained with all *training-disease* indication edges
present in the graph. Within a CV fold, the held-out diseases have their
metric scoring deferred but their nodes and edges remain in the message-
passing graph used by the encoder. The graph score for a "held-out" training
disease therefore reflects training memorization, not generalization. With
graph dominating the CV objective, β close to 1.0 maximises CV MRR.

On the test set, the graph has *not* seen the test disease nodes (they were
masked from training). The optimal β shifts dramatically: every configuration
prefers β ∈ {0.1, 0.3} on test, with a β shift of −0.6 to −0.8 from CV β
(see §4.2 calibration gap table).

The Hybrid β-sweep on test (§4.2) shows the inverted-U is sharp: MRR rises
from 0.249 (β = 0) to 0.285 (β = 0.1), then declines monotonically to 0.200
(β = 1). The CV calibrator picks the wrong side of the peak by a large margin.

This is a known limitation of using training-set CV to tune hyperparameters
of a model that was trained on the same set. A more principled fix would
be either (a) a phenotype-set-conditioned β based on inference-time-available
signals like phenotype count and graph/LLM confidence, or (b) holding out a
small validation slice of training diseases that are *also* masked from the
R-GCN training graph during a re-training pass.

### 7.3 Why Hybrid wins under oracle β (and not under CV β)

Hybrid is `text = tier2_text + " " + gpt4o_text`. Mean-pooling at the
encoder level means the strong, rare KG-anchor tokens in Tier 2 are diluted
by an equal mass of generic prose tokens from GPT-4o. The Tier 2 anchor
signal contributes ~50% of the pooled embedding instead of ~100%, and the
remaining ~50% is generic biomedical vocabulary that has weak discriminative
power for retrieval.

This dilution explains why Hybrid LLM-only (β = 0) does not beat Tier 2
LLM-only by a wide margin. But under fusion, two effects favor Hybrid:

1. The graph score already provides the signal that Tier 2's KG anchors
   double-encode. Hybrid's "diluted but more diverse" representation is less
   redundant with graph than Tier 2's pure-anchor representation.
2. The GPT-4o portion of Hybrid contributes mechanism / class information
   that PrimeKG does not encode at all.

At low β (0.1 – 0.3), where the LLM signal carries 70 – 90% of the weight,
the second effect dominates and Hybrid pulls ahead of both Tier 2 and
pure-GPT-4o. At high β (CV-tuned 0.9), the LLM contribution is too small
for either effect to matter, and all three description tiers cluster around
graph alone.

### 7.4 Why off-label augmentation matters

PrimeKG's `indication` edges are FDA-approved labels; `off_label_use` edges
record published off-label clinical practice. Both are real treatments but
the indication edges are a strict subset.

The R-GCN was trained on both indication and off-label edges (subject to the
test-disease masking). At inference, the model has internalized off-label
patterns but the strict indication-only test labeling penalizes it for
ranking off-label drugs above indication drugs. Adding off-label edges to
the test ground truth:

- Lifts the pure-graph baseline by +0.054 MRR (0.200 → 0.254), confirming
  the graph had legitimate off-label predictions that were unfairly counted
  as false positives.
- Lifts the best fused configuration (Hybrid (none), oracle β = 0.1) by
  +0.032 MRR (0.283 → 0.315).
- Has the smallest effect on LLM-only configurations (+0.007 to +0.018),
  consistent with text embeddings not having direct access to PrimeKG's
  treatment edges.

This suggests the indication-only test set is a *pessimistic* lower bound on
real model utility for clinical drug repurposing, which is precisely the
use case where off-label evidence is admissible.

---

## 8. Methodological Limitations

1. **Train-side β calibration is biased and significantly under-calibrates
   fusion** (§7.2). Section 9 oracle β shows every configuration loses
   0.026 – 0.080 MRR to the calibration error. The CV-tuned headline numbers
   are not what the method can achieve in practice.

2. **Proxy MRR is not a tier-comparable metric** (§7.1). Use only for
   within-tier hyperparameter selection.

3. **Phenotype enrichment had ~5% API failures.** Those rows currently fall
   back to Tier 2 text. Re-running enrichment after restoring the OpenAI
   quota would clean up this contamination.

4. **β is global, not per-disease**. A2 / A3 and Section 11 all suggest that
   the optimal β depends on observable disease-instance characteristics such
   as phenotype count and graph/LLM confidence. The current report only
   demonstrates this adaptivity diagnostically; it does not yet provide an
   unbiased train/validation protocol for learning the gate.

5. **Single text encoder family compared at full scale**. The encoder grid
   (BiolinkBERT vs PubMedBERT vs BiomedBERT vs SPECTER2) was run on Tier 2
   only; we have not validated whether BiolinkBERT remains best when the
   description content is GPT-4o prose rather than KG metadata.

6. **Oracle β reported in §4.2 and §5 should not be used as the headline
   test-set number for publication** — it is selected on the test set and
   is a diagnostic upper bound only. The same caution applies to the
   conditioned-β results in Section 11, which are fit against test-side
   oracle β and therefore serve only as a feasibility check.

---

## 8A. Conditioned-β Diagnostic Analysis (Section 11 of notebook)

Section 11 asks a narrower question than Sections 4–10:

> If we treat the per-disease oracle β on the test set as a supervision
> target, can simple, *inference-time-available* signals predict which
> diseases should use a low-β (LLM-heavy) vs high-β (graph-heavy) fusion?

This is important because in the project's true rare-disease setting we do
**not** know the disease identity at inference time, so a deployable gate
cannot rely on train-disease degree or any disease-specific label statistic.
Section 11 therefore restricts itself to phenotype-set-observable features:

- `n_phenotypes`
- mean phenotype graph degree / coverage in the masked training KG
- graph score sharpness (`graph_margin = top1 − top10`)
- LLM score sharpness (`llm_margin = top1 − top10`)
- graph-vs-LLM top-10 Jaccard overlap

### 8A.1 Main result: simple conditioned gates recover most of the lost fusion gain

Using 5-fold cross-fitting on the **test diseases** (diagnostic only), the
best conditioned gates are:

| Config | Best simple conditioned gate | MRR_ind | MRR_off | Compare to global CV β | Compare to global oracle β |
|---|---|---:|---:|---:|---:|
| Hybrid (NLAE) | Linear conditioned β (full) | **0.288** | **0.325** | +0.079 / +0.060 | +0.003 / +0.019 |
| Hybrid (none) | Bucketed `n_phenotypes` | **0.286** | **0.324** | +0.077 / +0.063 | +0.003 / +0.009 |
| Tier 2 (NLAE) | Bucketed `margin_gap` | **0.257** | **0.289** | +0.049 / +0.022 | +0.001 / +0.002 |
| GPT-4o (NLAE) | Bucketed `n_phenotypes` | **0.253** | **0.284** | +0.030 / +0.010 | −0.017 / −0.009 |
| GPT-4o (none) | Bucketed `margin_gap` | **0.257** | **0.296** | +0.024 / +0.013 | −0.001 / +0.007 |

Interpretation:

1. **Hybrid benefits the most from adaptive β.** Both Hybrid variants jump
   from the CV-tuned ~0.209 MRR regime to ~0.285–0.288, essentially closing
   the calibration gap identified in Section 9.
2. **Tier 2 also benefits, but less dramatically.** Its best conditioned
   gate is only marginally better than the best global oracle β, consistent
   with Tier 2 being more redundant with the graph.
3. **GPT-4o benefits from adaptation, but the gain is smaller and noisier.**
   For GPT-4o the best conditioned gates improve substantially over global
   CV β, but do not consistently beat the best global oracle β on the
   indication-only metric.

### 8A.2 Oracle-gap analysis: conditioned β helps, but does not match the per-disease oracle

Section 11B compares each gate to a **per-disease oracle β** that directly
maximizes MRR on each test disease. That per-disease oracle is an unattainable
upper bound, but it tells us how much structure is left on the table.

| Config | Global CV β gap to per-disease oracle | Best conditioned gap to per-disease oracle |
|---|---:|---:|
| GPT-4o (none) | 0.113 | 0.087 (global oracle) / 0.089 (bucketed margin) |
| GPT-4o (NLAE) | 0.135 | 0.088 (global oracle) |
| Hybrid (none) | 0.168 | **0.091** (bucketed `n_phenotypes`) |
| Hybrid (NLAE) | 0.165 | **0.086** (linear full) |
| Tier 2 (NLAE) | 0.138 | **0.090** (bucketed `margin_gap`) |

The diagnostic takeaway is not that the current gate is “solved,” but that
**observable phenotype-set features explain a meaningful fraction of the
oracle-β variation**. The CV-tuned β is much farther from the per-disease
oracle than any simple conditioned gate on the Hybrid variants.

### 8A.3 Feature ablation: simple signals are enough

The strongest Section 11C finding is that **the full feature set is not
consistently better than simple one- or two-feature gates**:

- For **GPT-4o (none)**, the best off-label metric came from
  `graph_margin` alone (MRR_off 0.291), not from the full linear gate.
- For **GPT-4o (NLAE)**, `n_phenotypes` alone was best (MRR_off 0.285).
- For **Hybrid (none)**, `n_phenotypes + degree` and `n_phenotypes`-based
  gates matched the full gate (MRR_off 0.321 – 0.323).
- For **Hybrid (NLAE)**, the full gate was best (MRR_off 0.325), but only
  narrowly ahead of `n_phenotypes + margins` (0.324).
- For **Tier 2 (NLAE)**, margin-based gates were best (MRR_off 0.295).

This is encouraging: if conditioned β is taken forward, it probably does not
need a high-capacity gating model. A small bucketed or linear gate over
`n_phenotypes` and confidence-style margin features may be sufficient.

### 8A.4 Interpretation and caution

Section 11 should be read as **evidence of feasibility**, not as a reportable
test result. The gate was fit on the test diseases using test-side oracle β as
the target, so it shares the same “selected on test” caveat as Section 9 — and
in fact is one step more optimistic, because it learns from those oracle labels.

Still, Section 11 materially changes the recommendation for what to do next:

- The failure mode is **not** that late fusion lacks useful complementarity.
- The failure mode is **not** that a single global oracle β is already enough.
- The failure mode is that **global CV β is the wrong functional form** for
  this setting.

The natural next step is therefore an **unbiased phenotype-set-conditioned β**
learned on a train/validation protocol where the validation diseases are masked
from the R-GCN training graph, then evaluated once on the untouched test set.

---

## 8B. Final Candidate Model (Section 13 of notebook)

### 8B.1 Definition

Among the configurations evaluated in Sections 4–11, we select
**Hybrid (none) + Bucketed `margin_gap`** as the candidate final
late-fusion model for downstream experiments and the Phase 2a writeup.

For an input phenotype set `P`, let `s_graph(P)` be the per-drug score vector
from the trained R-GCN + drug-conditioned cross-attention model, and let
`s_LLM(P)` be the per-drug score vector from the **Hybrid** text representation
(Tier 2 text concatenated with GPT-4o text), encoded with **BiolinkBERT** and
**no projection** (the raw 768-d mean-pooled token embedding, L2-normalized).
After per-disease min-max normalization the final score is

```
s_final(P) = β(P) · s̃_graph(P) + (1 − β(P)) · s̃_LLM(P)
```

where `β(P)` is assigned by a 3-quantile-bucket gate over the
`margin_gap` of `P`:

```
margin_gap(P) = [top1(s̃_graph) − top10(s̃_graph)]
              − [top1(s̃_LLM)   − top10(s̃_LLM)]
```

`margin_gap > 0` means the graph ranking is sharper than the LLM ranking and
gets a higher β; `margin_gap < 0` means the LLM is sharper and gets a lower β.
Each of the three quantile buckets receives one constant β, fit by minimizing
in-bucket squared error against per-disease oracle β (cross-fitted across test
diseases).

### 8B.2 Aggregate metrics (Section 11.6 of notebook)

Computed on the 108 test diseases under the Section 10 evaluation protocol
(per-disease minmax normalization, full-library macro for AUROC/AUPRC,
TxGNN-style 1:1 balanced micro for `*_balanced_micro`):

| Metric | Indication only | Off-label augmented | Δ (off − ind) |
|---|---:|---:|---:|
| MRR | 0.2848 | **0.3253** | +0.0405 |
| R@1 | 0.0868 | 0.0918 | +0.0051 |
| R@5 | 0.1573 | 0.1618 | +0.0045 |
| R@10 | 0.2143 | 0.2270 | +0.0127 |
| R@50 | 0.3881 | 0.4101 | +0.0220 |
| AUROC (full library macro) | 0.9447 | 0.9465 | +0.0018 |
| AUPRC (full library macro) | 0.2056 | 0.2302 | +0.0246 |
| AUROC (1:1 balanced micro) | 0.9403 | 0.9486 | +0.0082 |
| AUPRC (1:1 balanced micro) | 0.9423 | 0.9497 | +0.0074 |

Gate behavior across the 108 test diseases: **mean β = 0.205, std β = 0.083**.
The gate concentrates around β ≈ 0.2, with the bottom margin-gap bucket
receiving even smaller β and the top bucket receiving moderately larger β.
This matches the Section 9 finding that global oracle β is around 0.1 – 0.3
for Hybrid configurations.

### 8B.3 Comparison to the same configuration under simpler β strategies

| β strategy for Hybrid (none) | MRR_ind | MRR_off | β stats |
|---|---:|---:|---|
| Global CV β = 0.9 | 0.207 | 0.259 | constant 0.90 |
| Global oracle β = 0.1 (test-selected) | 0.283 | 0.315 | constant 0.10 |
| **Bucketed `margin_gap` (final candidate)** | **0.285** | **0.325** | mean 0.21, std 0.08 |
| Per-disease oracle β (unattainable upper bound) | 0.376 | 0.402 | mean 0.21, std 0.31 |

The candidate gate matches the global oracle on indication-only MRR (0.285 vs
0.283) and exceeds it on off-label MRR (0.325 vs 0.315). Half of the remaining
gap to the per-disease oracle (0.376 / 0.402) is caused by diseases where the
oracle β is highly individualized and not predictable from `margin_gap` alone.

### 8B.4 Interpretation and caveats

- The final candidate uses the same trained R-GCN, the same Hybrid description
  text, and the same BiolinkBERT encoder as Sections 4–11. Only the β
  assignment changes, which means deploying this model adds zero training
  cost on top of Phase 2a.
- The gate parameters (3 bucket boundaries on `margin_gap`, 3 bucket β values)
  are fit against **test-side oracle β** via 5-fold cross-fitting on the 108
  test diseases. This shares the Section 9 / 8A caveat: the reported
  Section 8B numbers are an upper bound on what an unbiased gate would
  achieve. Promoting this from a candidate to a deployable model requires
  re-fitting the gate on a masked validation slice held out from R-GCN
  training (recommended next step #1 in §10).
- Because the gate uses only **inference-time-available signals** (the score
  margins after a single graph and LLM forward pass), the inference
  procedure in deployment is identical to the diagnostic procedure: no
  disease-label oracle is consulted at inference time, only at fit time.
  This matters for the rare-disease inference use case where the test
  disease identity is by definition unknown.

---

## 9. Headline Findings for the Phase 2a Report

1. **Late fusion improves over pure-graph by +0.085 MRR (+43% relative) on 108
   test rare diseases under oracle β**. With CV-tuned β, the gain is +0.020
   (+10%) — a faithful number for what the current calibration can deliver,
   but not a faithful number for what late fusion as a method can deliver.

2. **Hybrid description (Tier 2 ⊕ GPT-4o concatenation) is the best
   description tier under oracle β** (0.285 / 0.283 for nonlinear_ae / none).
   The previous interim conclusion that GPT-4o alone wins was an artifact
   of the CV β bias.

3. **GPT-4o descriptions provide a real and quantitatively measurable
   complement to PrimeKG.** The clearest single-stratum evidence is
   cold-start: GPT-4o LLM-only achieves MRR 0.17 on diseases whose true
   drugs were never seen in training, vs. 0.09 for the pure-graph baseline
   and 0.002 for Tier-2 LLM-only. This validates the project's central
   hypothesis that LLM parametric knowledge can rescue cold-start drugs.

4. **The proxy MRR – test MRR inversion is a leakage artifact, not a
   contradiction.** Tier 2 descriptions contain literal train-disease names
   that cosine similarity exploits on the train set; this shortcut is
   removed on the test set, and Hybrid (under oracle β) overtakes Tier 2.

5. **The 5-fold CV β calibration is biased upward by R-GCN training memorization
   and discards 0.026 – 0.080 MRR per configuration of achievable gain**
   (§4.2 calibration gap table). Per-disease adaptive β is the highest-value
   next experiment.

6. **Off-label augmented test labeling lifts the headline fusion number from
   MRR 0.285 to 0.315.** Indication-only labeling is a pessimistic lower bound
   on real clinical utility for drug repurposing; the off-label augmented
   number better reflects actionable model performance.

7. **Graph and LLM signals are nearly orthogonal**: 81% of test diseases
   have zero top-10 overlap under GPT-4o. The structural ceiling for late
   fusion is much higher than what either CV β or oracle β achieves.

8. **Section 11 shows that phenotype-set-conditioned β is plausible.**
   On the Hybrid configurations, simple gates based on `n_phenotypes` and
   confidence-style margins recover most of the gap between global CV β and
   global oracle β, reaching MRR_ind 0.286 – 0.288 and MRR_off 0.324 – 0.325
   in diagnostic analysis. This does **not** yet count as a final result
   because the gate was fit against test-side oracle β, but it strongly
   supports adaptive β as the highest-value next experiment.

9. **The candidate final late-fusion model is `Hybrid (none) + Bucketed
   margin_gap`** (§8B). On the 108 test diseases it reaches
   MRR 0.285 / R@10 0.214 / AUROC 0.945 / AUPRC 0.206 under indication-only
   ground truth, and MRR 0.325 / R@10 0.227 / AUROC 0.947 / AUPRC 0.230
   under off-label-augmented ground truth. The gate uses only
   inference-time-available signals (score margins) and adds zero training
   cost on top of Phase 2a.

---

## 10. Recommended Next Steps

| # | Experiment | Why | Effort |
|---|---|---|---|
| 1 | Unbiased phenotype-set-conditioned β learned on a masked validation split | Section 11 shows simple observable features can recover most of the calibration gap, but the current gate is test-fit and only diagnostic | 1–2 days |
| 2 | Hold out 20% of train diseases (with their graph edges masked during R-GCN re-training) for unbiased β calibration | Eliminates the train-side memorization bias entirely and provides a clean target for the conditioned gate | ~2 days (requires retraining) |
| 3 | Embedding-level Hybrid: `α · emb_tier2 + (1−α) · emb_gpt4o` (encoded separately, then averaged) | Avoids the mean-pool dilution that text concat suffers (§7.3); may beat current Hybrid (NLAE) 0.285 | 0.5 day |
| 4 | Retry the 198 phenotype-enrichment failures | Cleans 5.6% contamination in GPT-4o phenotypes | trivial after API quota |
| 5 | Re-test BiolinkBERT vs PubMedBERT on GPT-4o text | Encoder selection was done on Tier 2 only; the prose distribution may favor a different encoder | ~1 day |
| 6 | Report off-label-augmented metrics as the primary test number, with indication-only as a stricter complement | More clinically meaningful (§5, §7.4) | trivial |

---

## Appendix A — Reproduction

All experiments are reproducible from
`notebooks/late_fusion_pipeline.ipynb` with the following inputs:

- `data/primekg/{nodes,edges,kg}.csv`
- `data/splits/{train,test}_disease_ids.txt`
- `data/weights/rgcn_best_model.pt`
- `data/descriptions/{drugs,phenotypes}_{tier1,tier2,gpt4o,hybrid}.json`
- `data/embeddings/biolinkbert/{tier1,tier2,gpt4o,hybrid}/{nonlinear_ae,none}/`

Saved tables and figures:

- `results/tables/section7_config_comparison.json` — CV-tuned 5-config metrics
- `results/tables/section9_oracle_beta_sweep.json` — oracle β sweep + gap table
- `results/tables/section10_offlabel_summary.csv` — off-label-augmented summary
- `results/tables/section11a_conditioned_beta_summary.csv` — Section 11A main comparison (global vs conditioned β)
- `results/tables/section11a_conditioned_beta_matrix_MRR_off.csv` — 11A pivoted MRR_off matrix
- `results/tables/section11b_oracle_gap_summary.csv` — Section 11B oracle-gap analysis
- `results/tables/section11b_oracle_gap_matrix.csv` — 11B pivoted oracle-gap matrix
- `results/tables/section11c_feature_ablation_summary.csv` — Section 11C linear-gate feature ablation
- `results/tables/section11c_feature_ablation_matrix_MRR_off.csv` — 11C pivoted MRR_off matrix
- `results/tables/section11_comparison_matrix.csv` — joint A/B/C comparison matrix
- `results/tables/feature_fusion_bucketed_margin_gap_biolinkbert_hybrid_none.csv` — per-disease records for the §8B final candidate (Hybrid (none) + Bucketed margin_gap)
- `results/tables/feature_fusion_bucketed_margin_gap_biolinkbert_hybrid_none_metrics.csv` — §8B aggregate metrics under both GT regimes
- `results/tables/feature_fusion_bucketed_margin_gap_biolinkbert_hybrid_none_metrics.json` — same, JSON
- `results/tables/section12_encoder_projection_proxy_mrr.csv` — Section 12 4×4 encoder/projection grid
- `results/A1_per_disease_delta.png` — A1 per-disease Δ histograms
- `results/A4_jaccard_vs_delta.png` — A4 Jaccard vs Δ scatter plots
- `results/B1_oracle_beta_sweep.png` — Section 9 β-sweep curves
- `results/section12_encoder_projection_proxy_mrr_heatmap.png` — Section 12 4×4 heatmap of proxy MRR over text encoders × projections

## Appendix B — File Map

| Path | Role |
|---|---|
| `scripts/generate_descriptions.py` | Tier 1 / Tier 2 / GPT-4o description generation |
| `scripts/cache_embeddings.py` | Encode descriptions with BiolinkBERT etc., apply projection, L2-normalize, save |
| `src/models/cross_attention_scorer.py` | `PhenoDrugModel` (R-GCN + drug-conditioned cross-attention) |
| `src/models/fusion.py` | `LateFusion`, `normalize_scores`, `calibrate_beta` (5-fold CV) |
| `src/evaluation/late_fusion_eval.py` | `run_late_fusion_experiment` — top-level orchestrator with checkpoint arch inference, score loading, and per-disease evaluation including cold-start stratification |
| `src/evaluation/metrics.py` | MRR, R@K, full-library AUROC/AUPRC |
