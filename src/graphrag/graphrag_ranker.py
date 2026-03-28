"""Chain-of-thought LLM prompting for final drug ranking from GraphRAG paths.

TODO: Implement CoT prompting pipeline:
  - Input: serialized KG paths + phenotype context
  - Output: ranked drug candidates with reasoning traces
"""

from __future__ import annotations
