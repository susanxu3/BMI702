"""End-to-end GraphRAG pipeline for all 108 test diseases (Phase 4).

Pipeline steps:
  1. Build adjacency maps from training KG (test diseases already masked)
  2. For each test disease's phenotype set:
     a. Extract paths via both path types (direct target + pathway-mediated)
     b. Prune paths (relevance + diversity, K-Paths style)
     c. Compute phenotype coverage count per drug
     d. Serialize paths as structured NL with entity descriptions
     e. Build CoT prompt and call GPT-4o for ranked drug list
  3. For drugs not in candidate pool D, assign text embedding cosine
     similarity as fallback score
  4. Evaluate: MRR, R@K, AUROC, AUPRC on full 7,957-drug ranking

Output: results/tables/graphrag_results.csv
"""

from __future__ import annotations
