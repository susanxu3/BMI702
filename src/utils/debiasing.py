"""Degree-based score correction for inference-time debiasing.

High-degree drugs (e.g., Stanolone, Ribavirin) are systematically over-ranked
by graph models. This module subtracts a degree-based bias term from raw scores.

TODO: Implement debiasing strategies:
  - Subtract log-degree bias: score_debiased = score_raw - beta * log(degree + 1)
  - Calibrate beta on validation set
"""

from __future__ import annotations
