"""Cascade baseline: Jaccard phenotype matching + TxGNN reranking (MRR ~ 0.121).

TODO: Implement two-stage cascade:
  1. Jaccard similarity over phenotype sets to find most similar train diseases
  2. Use TxGNN scores from matched diseases to rank drugs
"""

from __future__ import annotations
