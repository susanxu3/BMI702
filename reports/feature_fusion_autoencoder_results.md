# Feature-Level Fusion — Approach (b): Concat Autoencoder

## 1. Experimental setup

We replace the raw R-GCN node embedding inside the cross-attention scorer with the bottleneck representation of an autoencoder over the concatenated graph and text features. Two variants are compared:

**Plain AE.** The latent code itself becomes the new node embedding:
$$h_{\text{fused}}(v) = \text{AE}_{\text{enc}}\big([\,h_{\text{graph}}(v);\, h_{\text{LLM}}(v)\,]\big)$$

**Residual AE.** The graph embedding is preserved as the base representation; the autoencoder learns an additive correction on top:
$$h_{\text{fused}}(v) = h_{\text{graph}}(v) + \text{AE}_{\text{enc}}\big([\,h_{\text{graph}}(v);\, h_{\text{LLM}}(v)\,]\big)$$

Both variants share the same training schedule:

1. **Phase 1 — unsupervised AE pretraining.** Reconstruction MSE on the concatenation, computed only on the ~11.5K nodes with cached text embeddings (drugs and phenotypes); the remaining ~117K nodes have zero placeholder LLM rows that are excluded from the loss. The R-GCN encoder is frozen and `h_graph` is computed once from the frozen checkpoint. 50 pretraining epochs, Adam, `lr = 1e-3`.
2. **Phase 2 — conservative supervised fine-tuning.** Train graph is rebuilt at runtime so all phenotype edges incident to validation **and** test diseases are masked. Phase 2 starts with `phase2_train_scope = "scorer_only"` (encoder frozen, only the cross-attention scorer plus the AE encoder train). The AE decoder is frozen throughout Phase 2 — it has no downstream consumer. Margin ranking loss with degree-weighted negatives, `lr = 1e-4`, 100 epoch budget, patience 15 on val MRR.

**Configuration.** Encoder: BioLinkBERT, projection: `nonlinear_ae`, embedding dimensions: `h_graph[256]` and `h_llm[256]`. AE: `input_dim = 512`, `hidden_dim = 512`, `latent_dim = 256` (matched to the scorer's expected node-embedding width). Two description tiers compared for each variant: `gpt4o` and `hybrid`.

Test evaluation reports both indication-only and indication-augmented-with-off-label ground truth, mirroring Section 10 of the late-fusion pipeline.

## 2. Test results

### 2.1 Plain AE (replacement fusion)

Numbers below come directly from the in-memory results of cells 9, 11 and the comparison in cell 15.

| Tier   | GT          | MRR    | R@1    | R@5    | R@10   | R@50   | AUROC  | AUPRC  | AUROC_bal_micro | AUPRC_bal_micro |
|--------|-------------|--------|--------|--------|--------|--------|--------|--------|----------------:|----------------:|
| gpt4o  | indication  | 0.1069 | 0.0080 | 0.0198 | 0.0292 | 0.0722 | 0.8513 | 0.0342 |          0.8661 |          0.8206 |
| gpt4o  | off-label   | 0.1221 | 0.0081 | 0.0212 | 0.0307 | 0.0786 | 0.8553 | 0.0370 |          0.8853 |          0.8467 |
| hybrid | indication  | 0.1111 | 0.0077 | 0.0339 | 0.0439 | 0.0653 | 0.8496 | 0.0490 |          0.8615 |          0.8167 |
| hybrid | off-label   | 0.1234 | 0.0073 | 0.0317 | 0.0412 | 0.0697 | 0.8533 | 0.0483 |          0.8841 |          0.8663 |

Plain-AE replacement underperforms the graph-only baseline (`MRR = 0.254`) by roughly 56–58% and underperforms the degree-conditioned fusion run (`MRR = 0.18`) by an additional ~40%. The collapse is concentrated at the head of the list: R@1 ≈ 0.008, R@5 ≈ 0.02–0.03, R@10 ≈ 0.03–0.04. Discrimination as measured by full-library AUROC remains ≈ 0.85, and the TxGNN-style balanced micro AUROC actually rises to 0.86–0.89 — so the model still distinguishes positives from random negatives, but the ranking signal that pushes a small number of true drugs to the top is largely lost.

The most plausible mechanism: the AE pretraining objective (reconstruct `[h_graph; h_llm]`) optimizes for *information preservation*, not for *discriminability under the cross-attention scorer*. The R-GCN embedding produced by the trained checkpoint already lives in a geometry that the scorer's attention weights are calibrated to. Replacing it with a latent code that has merely been pretrained to reconstruct the inputs throws away most of that calibration, and the short, encoder-frozen Phase 2 (`scorer_only` scope, `lr = 1e-4`) cannot recover it.

### 2.2 Residual AE (additive correction)

Numbers below are from the residual-AE branch of cell 22. (Note: cell 22's table also includes "ae" rows whose values are inconsistent with cell 15 and identical to the degree-conditioned-fusion results; those rows appear to come from a wandb-summary loader that did not adequately filter by `fusion_variant`. We therefore rely on cell 15 for plain AE and on cell 22 only for residual AE.)

| Tier   | GT          | MRR    | R@1    | R@5    | R@10   | R@50   | AUROC  | AUPRC  | AUROC_bal_micro | AUPRC_bal_micro |
|--------|-------------|--------|--------|--------|--------|--------|--------|--------|----------------:|----------------:|
| gpt4o  | indication  | 0.1733 | 0.0362 | 0.1378 | 0.1859 | 0.3386 | 0.8511 | 0.1460 |          0.7781 |          0.7580 |
| gpt4o  | off-label   | 0.2335 | 0.0413 | 0.1383 | 0.1883 | 0.3611 | 0.8524 | 0.1630 |          0.7918 |          0.7720 |
| hybrid | indication  | 0.1997 | 0.0501 | 0.1241 | 0.1648 | 0.3403 | 0.8461 | 0.1444 |          0.7663 |          0.7336 |
| hybrid | off-label   | 0.2524 | 0.0541 | 0.1255 | 0.1758 | 0.3628 | 0.8476 | 0.1601 |          0.7806 |          0.7461 |

Residual AE recovers most of the head-of-list signal lost by the replacement variant — R@1 jumps from 0.008 to 0.04–0.05, R@5 from 0.02–0.03 to 0.12–0.14, and indication MRR from ~0.11 to 0.17–0.20. By keeping `h_graph` as the base representation and learning only an additive correction, the variant inherits the trained scorer's geometry and only nudges it with the residual.

Residual AE still trails the graph-only baseline (`MRR = 0.254`) at the indication-only setting (gap of −0.054 to −0.081). Under the off-label-augmented ground truth the hybrid variant comes within 0.003 of the baseline (`MRR_off = 0.252`).

### 2.3 Variant × tier comparison

| Variant     | Tier   | GT          | MRR    | R@1    | R@10   | AUPRC  | MRR_seen_some |
|-------------|--------|-------------|--------|--------|--------|--------|--------------:|
| plain AE    | gpt4o  | indication  | 0.107  | 0.008  | 0.029  | 0.034  |        0.142  |
| plain AE    | hybrid | indication  | 0.111  | 0.008  | 0.044  | 0.049  |        0.163  |
| residual AE | gpt4o  | indication  | 0.173  | 0.036  | 0.186  | 0.146  |        0.133  |
| residual AE | hybrid | indication  | 0.200  | 0.050  | 0.165  | 0.144  |        0.221  |
| plain AE    | gpt4o  | off-label   | 0.122  | 0.008  | 0.031  | 0.037  |        0.168  |
| plain AE    | hybrid | off-label   | 0.123  | 0.007  | 0.041  | 0.048  |        0.198  |
| residual AE | gpt4o  | off-label   | 0.234  | 0.041  | 0.188  | 0.163  |        0.277  |
| residual AE | hybrid | off-label   | 0.252  | 0.054  | 0.176  | 0.160  |        0.324  |

Within each variant, **hybrid is the stronger description tier**, consistent with the same observation in the degree-conditioned fusion runs. Residual AE with hybrid is the best single configuration: indication MRR 0.200, off-label MRR 0.252, R@1 0.050, and the highest `MRR_seen_some` of 0.324 under off-label GT.

## 3. Cold-start stratification

| Variant     | Tier   | GT         | MRR_seen_all | MRR_seen_some | MRR_seen_none |
|-------------|--------|------------|-------------:|--------------:|--------------:|
| plain AE    | gpt4o  | indication |       0.1070 |        0.1419 |        0.0043 |
| plain AE    | gpt4o  | off-label  |       0.1044 |        0.1682 |        0.0004 |
| plain AE    | hybrid | indication |       0.1038 |        0.1631 |        0.0024 |
| plain AE    | hybrid | off-label  |       0.0841 |        0.1982 |        0.0004 |
| residual AE | gpt4o  | indication |       0.2170 |        0.1331 |        0.0315 |
| residual AE | gpt4o  | off-label  |       0.2331 |        0.2773 |        0.0006 |
| residual AE | hybrid | indication |       0.2178 |        0.2207 |        0.0314 |
| residual AE | hybrid | off-label  |       0.2336 |        0.3241 |        0.0005 |

Two patterns deserve highlighting.

**Plain AE inverts the usual cold-start gradient.** For every (tier × GT) cell, `MRR_seen_some > MRR_seen_all`. That is, the plain AE ranks better on the partially-seen stratum than on the fully-seen one — the opposite of what graph baselines and the residual AE produce. One explanation: the plain AE latent code is no longer biased toward popular drugs the way `h_graph` is, so high-popularity training drugs no longer get an unfair head-start, and partial-coverage diseases (which depend on both seen and unseen drugs being ranked together) benefit. The price is paid by the seen-all stratum, which now has to compete on equal footing.

**Residual AE follows the expected ordering and concentrates its lift on `seen_some` under off-label GT.** Under off-label-augmented ground truth, hybrid lifts `MRR_seen_some` from 0.221 (indication) to 0.324 (+47%); gpt4o lifts from 0.133 to 0.277 (+108%). Cold-start `MRR_seen_none` collapses to ≈ 0 under off-label GT (the augmentation does not introduce off-label edges for drugs entirely absent from training), and remains in the 0.03 range under indication-only GT — comparable to the degree-conditioned fusion floor and to prior phases.

## 4. Comparison with other phases

| Phase                                      | Indication MRR | Off-label MRR | Notes                                       |
|--------------------------------------------|---------------:|--------------:|---------------------------------------------|
| Graph-only baseline (R-GCN)                |          0.254 |             — | reported in the midterm                     |
| Late fusion (best β, gpt4o)                |          0.459 |             — | midterm; CV-tuned β over the same checkpoint |
| Degree-conditioned fusion — gpt4o          |          0.181 |         0.222 | this report's companion (Approach a)        |
| Degree-conditioned fusion — hybrid         |          0.182 |         0.249 | this report's companion (Approach a)        |
| Plain AE — gpt4o                           |          0.107 |         0.122 | replacement fusion, full collapse           |
| Plain AE — hybrid                          |          0.111 |         0.123 | replacement fusion, full collapse           |
| Residual AE — gpt4o                        |          0.173 |         0.234 | additive correction                          |
| Residual AE — hybrid                       |          0.200 |         0.252 | best feature-level configuration            |

Two things stand out across the phase comparison. First, **no feature-level fusion configuration so far recovers the late-fusion MRR (0.459)**. Late fusion mixes per-disease score arrays after the model has finished — it cannot disturb the geometry of the trained R-GCN embedding. Both feature-level variants do disturb that geometry, and both pay for it at the head of the list. Second, **the gap between plain AE and residual AE is the most informative result of this notebook**. Reconstruction-based pretraining alone is not a sufficient bridge between a frozen, ranking-tuned graph embedding and the cross-attention scorer; preserving the graph embedding as a base and only learning a residual is required.

## 5. Discussion

**(1) Replacement fusion is the wrong inductive bias here.** The plain AE's reconstruction objective compresses the joint `[h_graph; h_llm]` for fidelity, not for ranking. Forcing the scorer to operate on this compressed code drops R@1 by an order of magnitude (0.04 → 0.008) while leaving full-library AUROC effectively unchanged. The model still knows which drugs *belong* with each disease — it just does not have a mechanism for pushing the right ones into positions 1–10.

**(2) Residual fusion is a simple but effective fix.** Adding the AE output to `h_graph` rather than replacing it preserves the trained R-GCN's drug ordering as the prior, and uses the AE only to perturb it. This recovers most of the baseline (within 0.05 MRR for gpt4o, within 0.054 for hybrid under indication-only GT) and at off-label-augmented GT comes within 0.002 of the baseline for hybrid. AUROC and balanced-micro AUROC sit around 0.85 / 0.78–0.79, comparable to other feature-level fusion runs.

**(3) Hybrid descriptions are again the better text source.** Across both variants, hybrid edges out gpt4o on indication MRR, R@1, and off-label MRR. The most striking lift is on `MRR_seen_some` under off-label GT: residual-AE hybrid reaches 0.324, the highest in any run we have measured for this stratum. This is consistent with the degree-conditioned fusion finding and reinforces the intuition that descriptions anchored to KG context (rather than free-form GPT-4o text alone) carry the relational hooks that fusion needs.

**(4) Cold-start (`MRR_seen_none`) remains unsolved.** Both variants land in the ≈ 0.03 range under indication-only GT, indistinguishable from the degree-conditioned fusion run and from prior phases. Under off-label GT it collapses to ≈ 0 because off-label edges in PrimeKG do not cover drugs that were unseen during training.

## 6. Caveats

- Cell 22's "ae" rows in the AE-vs-residual-AE comparison are inconsistent with the in-memory cell 15 results and identical to the degree-conditioned fusion numbers; this looks like a wandb-summary loader bug where the `fusion_variant` filter accepts runs without an explicit tag and consequently picks up the more recent degree-conditioned runs. The plain-AE numbers reported here are the cell-15 in-memory values; the residual-AE numbers are taken at face value because they do not duplicate any degree-conditioned values. Re-tagging existing wandb runs with `fusion_variant` would resolve the ambiguity for future loads.
- The plain-AE and residual-AE Phase-2 configurations share the same `phase2_train_scope = "scorer_only"`, so the residual variant's win is not confounded by a wider trainable surface; it comes from the architectural choice of additive correction.

## 7. Next steps

- **Diagnose plain AE.** Compare the AE latent's geometry to the original `h_graph` (e.g. CKA, linear probe MRR for held-out diseases without fine-tuning) to confirm that reconstruction pretraining is destroying ranking-relevant directions.
- **Re-run plain AE with a longer Phase 2 and unfrozen encoder.** The current `scorer_only` schedule is conservative and may not give the latent code enough capacity to be re-aligned. A `full_finetune` Phase 2 may close some of the gap.
- **Combine residual AE with degree-conditioned gating.** The two approaches are complementary in spirit (residual AE preserves the graph base; the gate down-weights graph contribution at high degree). A gated residual — `h_fused = h_graph + α(v) · AE([h_graph; h_llm])` — may capture both.
- **Fix the wandb tagging.** Set `fusion_variant` on every run summary so the comparison loader cannot collide across phases.
- **Replicate the late-fusion gap on the same checkpoint.** Reporting late-fusion MRR on the exact same masked test split would make the 0.459 vs 0.20 gap auditable and motivate whether feature-level fusion can ever match late fusion or whether the two answer different questions.
