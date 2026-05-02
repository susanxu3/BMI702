# Feature-Level Fusion — Approach (a): Degree-Conditioned Gating

## 1. Experimental setup

We replace the raw R-GCN node embedding inside the cross-attention scorer with a degree-conditioned mix of graph and text representations:

$$
h_{\text{fused}}(v) = \alpha(v) \cdot h_{\text{graph}}(v) + (1 - \alpha(v)) \cdot h_{\text{LLM}}(v),
\quad \alpha(v) = \sigma(w \cdot \log(\deg(v) + 1) + b)
$$

The hypothesis underlying this design is that high-degree nodes accumulate enough message-passing signal to make `h_graph` reliable, while low-degree (cold-start) nodes benefit from the supplementary text signal — i.e. `α` should rise monotonically with `log_deg`, implying a positive `w`.

Two phases:

1. **Phase 1 — analytical gate calibration (no gradients).** Pair-level quartile partition by KG-structural drug `log_degree`, fixed-α sweep per quartile, closed-form linear regression on `(median log_deg, logit(α*))`. The R-GCN encode runs exactly once and `h_graph` is reused across all `4 × 11 = 44` evaluations. Gate `(w, b)` is frozen immediately after the regression.
2. **Phase 2 — supervised fine-tuning (gate frozen).** Margin ranking loss on training disease-drug pairs with degree-weighted negatives, plus a mask-restricted anchoring regularizer (`λ = 0.1`) over rows with cached text embeddings only.

**Configuration.** Encoder: BioLinkBERT, projection: `nonlinear_ae`, embedding dimensions: `h_graph[256]` and `h_llm[256]` (pre-projected and L2-normalized). Two description tiers compared in parallel runs:

- `gpt4o` — full GPT-4o-enriched entity descriptions.
- `hybrid` — KG-context-anchored descriptions enriched with GPT-4o.

Optimizer: Adam with `lr = 1e-4`, `weight_decay = 1e-5`, gradient accumulation 4 steps, gradient clip 1.0, batch size 512, AMP on. Phase 2 trains the scorer first (`phase2_train_scope = "scorer_only"`) with the encoder frozen. Validation: 10% of train diseases held out at runtime; early stopping on val MRR with patience 15.

Test evaluation uses the same fused scores under two ground-truth definitions: indication-only and indication-augmented-with-off-label, mirroring Section 10 of the late-fusion pipeline.

## 2. Phase-1 calibration

| Tier   | m_Q1 | α*_Q1 | m_Q2 | α*_Q2 | m_Q3 | α*_Q3 | m_Q4 | α*_Q4 | Fitted w | Fitted b |
|--------|------|-------|------|-------|------|-------|------|-------|---------:|---------:|
| gpt4o  | 6.68 | 1.0   | 8.18 | 0.9   | 8.60 | 0.9   | 8.86 | 0.7   |   −2.68  |  +24.66  |
| hybrid | 6.68 | 1.0   | 8.18 | 0.9   | 8.60 | 1.0   | 8.86 | 0.6   |   −1.94  |  +19.76  |

**Both tiers produced negative `w`.** That is, the optimal mix at the lowest-degree quartile is *pure graph* (α* = 1.0), and the gate down-weights the graph contribution as drug degree increases. This is the opposite of the design hypothesis, which expected `α` to rise with degree.

The implementation's tiered monotonicity check counts a violation when `α*_q > α*_{q+1} + 0.05`. By that criterion both tiers exhibit two violations (Q1→Q2 and Q3→Q4 for gpt4o; Q1→Q2 and Q3→Q4 for hybrid), placing them at or beyond the configured "warn-and-proceed" boundary. The gates were nonetheless frozen at the fitted values and Phase 2 proceeded; this is documented for transparency in interpretation.

## 3. Test results

All metrics are macro-averaged per disease. Both ground-truth variants are reported side by side; the indication-only column is the headline number for comparison with prior phases.

| Tier   | GT          | MRR    | R@1    | R@5    | R@10   | R@50   | AUROC  | AUPRC  | AUROC_bal_micro | AUPRC_bal_micro |
|--------|-------------|--------|--------|--------|--------|--------|--------|--------|----------------:|----------------:|
| gpt4o  | indication  | 0.1805 | 0.0398 | 0.1289 | 0.1723 | 0.3395 | 0.8590 | 0.1470 |          0.7870 |          0.7664 |
| gpt4o  | off-label   | 0.2223 | 0.0426 | 0.1297 | 0.1822 | 0.3636 | 0.8600 | 0.1620 |          0.8026 |          0.7793 |
| hybrid | indication  | 0.1818 | 0.0439 | 0.1368 | 0.1800 | 0.3441 | 0.8479 | 0.1501 |          0.7821 |          0.7611 |
| hybrid | off-label   | 0.2494 | 0.0494 | 0.1391 | 0.1949 | 0.3627 | 0.8495 | 0.1685 |          0.7960 |          0.7727 |

### 3.1 Comparison with prior baselines

The reported baseline R-GCN achieves `MRR = 0.254` (indication-only) on the same 108 test diseases. Both feature-level fusion runs land **below** that baseline at the indication-only setting (0.18 vs 0.25, a ~28% relative drop). Augmenting the test ground truth with off-label edges narrows the gap (gpt4o 0.222, hybrid 0.249) but does not surpass the graph-only baseline.

Per-task ranking statistics (AUROC ≈ 0.85, balanced micro AUROC ≈ 0.79–0.80) remain comfortably above random and consistent with prior work, indicating that the model still produces a sensible drug ordering — the degradation is concentrated at the head of the list (R@1, R@5, MRR), not at the global discrimination level.

### 3.2 Cold-start stratification

| Tier   | GT         | MRR_seen_all | MRR_seen_some | MRR_seen_none |
|--------|------------|-------------:|--------------:|--------------:|
| gpt4o  | indication |       0.2263 |        0.1389 |        0.0314 |
| gpt4o  | off-label  |       0.2382 |        0.2427 |        0.0005 |
| hybrid | indication |       0.2106 |        0.1752 |        0.0308 |
| hybrid | off-label  |       0.2396 |        0.3087 |        0.0005 |

Two patterns stand out:

- **Hybrid descriptions help the partially-seen stratum substantially.** Under the off-label-augmented ground truth, hybrid lifts `MRR_seen_some` from 0.175 to 0.309 (+76%), versus gpt4o lifting 0.139 to 0.243 (+75%). Hybrid is the stronger middle-popularity performer in absolute terms (0.309 vs 0.243).
- **Cold-start (`seen_none`) collapses to ≈ 0 under off-label GT.** Off-label edges in PrimeKG do not cover drugs that were unseen in training, so adding them inflates the denominator without contributing reachable positives, dragging the bucket toward zero. The indication-only floor (≈ 0.03) is the more informative number for cold-start performance: feature-level fusion as configured does not appreciably solve the cold-start problem.

## 4. Sanity ablations

Two short (20-epoch) runs were executed alongside the main GPT-4o experiment to isolate the contributions of fusion and gate calibration.

| Configuration                                          | Best val MRR | Test MRR (ind.) | Notes                         |
|--------------------------------------------------------|-------------:|----------------:|-------------------------------|
| Graph-only Phase 2 (`phase2_bypass_fusion = True`)     |       0.694 |          0.199 | scorer fine-tune over h_graph |
| Fixed α = 0.5 (skip Phase 1, w = b = 0)                |       0.634 |          0.159 | naive 50/50 mix               |
| Full pipeline (calibrated α)                           |       —     |          0.181 | 100 epoch, w = −2.68, b = +24.66 |

Two takeaways:

- A naive uniform mix (α = 0.5) is *worse* than graph-only fine-tuning (0.159 vs 0.199). Untargeted LLM injection actively contaminates the ranking signal at the head of the list.
- Calibrated-α Phase 1 partially recovers, scoring between the two ablations (0.181). The negative-`w` calibration concentrates LLM contribution where it apparently helps — the high-degree quartile — and matches the graph-only baseline at low degrees (α* = 1.0).

## 5. Validation–test gap

Validation MRR is computed on a 10% hold-out of training diseases, whose phenotype edges remain in the graph. Best-val MRRs landed in the 0.63–0.69 range across configurations, while test MRRs sit in 0.16–0.20. The roughly 4× gap reflects the structural difference between the two splits: validation diseases retain their phenotype neighborhoods in the encoder's input graph, whereas test diseases have all incident edges masked. The gap is consistent with prior phases of this project and is not specific to feature-level fusion — but its magnitude does indicate that the fine-tuning loop is leveraging in-graph phenotype information more than transferring abstract structural signals to truly held-out diseases.

## 6. Discussion

The two most consequential findings are:

**(1) The empirical gate is monotonically *anti*-correlated with degree.** The design assumed `w > 0`; the analytical calibration produced `w ≈ −2.7` (gpt4o) and `w ≈ −1.9` (hybrid). One mechanistic explanation is *graph popularity bias*: high-degree drugs are over-represented in the cross-attention scorer's ranking because they appear as positives for many training diseases, and shrinking their `h_graph` contribution (lower α) is a corrective. A second possibility is that low-degree nodes' R-GCN embeddings are dominated by their handful of neighbors and therefore behave more like high-confidence "specialist" features, while high-degree nodes' embeddings are diluted by averaging over many heterogeneous neighbors. Either way, the result challenges the cold-start motivation for this gate design and suggests that the real failure mode the gate corrects is popularity bias at the head, not signal sparsity at the tail.

**(2) Feature-level fusion under-performs the graph-only baseline at the head of the list.** R@1 (≈ 0.04), R@5 (≈ 0.13) and indication MRR (0.18) sit below the previously reported R-GCN-only numbers, while AUROC (0.85) and balanced AUPRC (0.77) remain healthy. Concretely the model still discriminates the right drugs from the wrong ones, but it is more conservative about which positive to put in the very top slots. The graph-only Phase-2 sanity ablation (test MRR 0.199) and the naive 50/50 mix (0.159) bracket the calibrated run (0.181), showing fusion as currently configured does not provide a free lunch: anything more aggressive than the post-calibration mix actively hurts top-of-list accuracy.

**(3) Hybrid descriptions are the better text source for partial-coverage diseases.** The clearest signal for the value of LLM augmentation is at `MRR_seen_some` under off-label GT, where hybrid substantially outperforms gpt4o (0.309 vs 0.243). For diseases whose true drugs are a mix of seen and unseen, hybrid descriptions appear to encode the relational hooks that the gate-frozen LLM head uses to lift the partially-seen positives.

**(4) Cold-start remains unsolved.** `MRR_seen_none ≈ 0.03` under indication-only GT, and effectively zero under off-label GT. Feature-level fusion does not, on its own, give cold-start drugs a functional ranking floor.

## 7. Next steps

- **Probe the negative-`w` regime directly.** Refit Phase 1 over a wider α grid (e.g. step 0.05 instead of 0.10) to localize the optimum more precisely, and stratify the sweep by phenotype-count buckets in addition to drug-degree buckets to test whether degree is even the right conditioning signal.
- **Compare against the late-fusion-only baseline at the same checkpoint.** Reported here is feature-level fusion vs graph-only. Adding the late-fusion line on the same test diseases would let us attribute the head-of-list regression to the architectural change rather than to the description tier.
- **Address popularity bias more directly.** If `w < 0` reflects a popularity correction, an inference-time log-degree debias (Phase 2b) might subsume the gate's contribution and recover top-1/top-5 accuracy without the val/test gap.
- **Re-check Phase-1 ground truth.** The pair-level MRR uses a single positive drug per pair as the target. Re-running calibration with the full disease-level positive set may produce a less degenerate α* curve, particularly at `Q4` where the current α* = 0.6–0.7 is the cluster of largest cross-tier disagreement.
- **Run Approach (b).** The autoencoder fusion uses the same checkpoint and embeddings — its head-of-list metrics would clarify whether the regression is intrinsic to feature-level fusion or specific to the degree-conditioned gating mechanism.
