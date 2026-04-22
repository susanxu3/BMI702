"""Chain-of-thought LLM ranking over GraphRAG candidate pool (Section 5.2.2).

Following Sanz et al. (2025), uses CoT prompting with GPT-4o to produce a
final ranked drug list with explicit reasoning traces.

For drugs NOT in the candidate pool D, ranking scores fall back to text
embedding cosine similarity from the late fusion pipeline, ensuring all
7,957 candidate drugs receive a score at evaluation time.
"""

from __future__ import annotations

COT_RANKING_PROMPT = """A patient presents with the following clinical phenotypes:
{phenotype_list}

Below are candidate drugs identified through biomedical knowledge graph traversal,
along with the mechanistic paths connecting them to the patient's phenotypes.
The "covers N patient phenotypes" annotation indicates how many of the patient's
phenotypes are mechanistically linked to each drug.

{evidence_context}

Instructions:
1. For each candidate drug, reason step-by-step about its therapeutic relevance
   to this patient's phenotype set, considering:
   - How many phenotypes it addresses (coverage)
   - Whether the mechanism of action is direct (protein target) or indirect (pathway)
   - Known clinical use and safety profile
2. Rank the top {top_k} drugs from most to least therapeutically relevant.
3. Return your answer as a JSON array of drug names in ranked order.

Think step by step, then provide your final ranking."""


def build_ranking_prompt(
    phenotype_names: list[str],
    evidence_context: str,
    top_k: int = 50,
) -> str:
    """Build the CoT ranking prompt for GPT-4o.

    Args:
        phenotype_names: Patient's HPO phenotype names.
        evidence_context: Serialized evidence from path_serializer.
        top_k: Number of top drugs to rank.

    Returns:
        Formatted prompt string.
    """
    return COT_RANKING_PROMPT.format(
        phenotype_list=", ".join(phenotype_names),
        evidence_context=evidence_context,
        top_k=top_k,
    )
