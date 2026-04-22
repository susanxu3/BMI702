"""Structured natural language serialization of KG paths for LLM input.

Converts pruned graph paths into structured text for chain-of-thought LLM
ranking prompts. Uses entity descriptions cached from the text embedding
pipeline (scripts/generate_descriptions.py).

Each drug in the candidate pool D is annotated with its phenotype coverage
count, providing a structural prior that drugs addressing multiple phenotypic
dimensions are more therapeutically compelling.
"""

from __future__ import annotations

from src.graphrag.subgraph_extractor import KGPath


def serialize_path(
    path: KGPath,
    node_names: dict[int, str],
    node_descriptions: dict[int, str] | None = None,
) -> str:
    """Convert a single KGPath to structured natural language.

    Args:
        path: A KGPath from subgraph extraction.
        node_names: Mapping from node index to human-readable name.
        node_descriptions: Optional mapping from node index to text description.

    Returns:
        Serialized path string.
    """
    pheno_name = node_names.get(path.phenotype_idx, str(path.phenotype_idx))
    protein_name = node_names.get(path.protein_idx, str(path.protein_idx))
    drug_name = node_names.get(path.drug_idx, str(path.drug_idx))

    if path.path_type == "direct_target":
        text = (
            f"Phenotype '{pheno_name}' is associated with protein '{protein_name}' "
            f"(via gene_phenotype), which is a target of drug '{drug_name}' "
            f"(via drug_protein)."
        )
    elif path.path_type == "pathway_mediated":
        pathway_name = node_names.get(path.pathway_idx, str(path.pathway_idx))
        text = (
            f"Phenotype '{pheno_name}' is associated with protein '{protein_name}' "
            f"(via gene_phenotype), which participates in pathway '{pathway_name}' "
            f"(via protein_pathway), which is targeted by drug '{drug_name}' "
            f"(via drug_pathway)."
        )
    else:
        text = f"Unknown path type: {path.path_type}"

    # Append description if available
    if node_descriptions and path.drug_idx in node_descriptions:
        text += f" Drug description: {node_descriptions[path.drug_idx]}"

    return text


def serialize_drug_evidence(
    drug_idx: int,
    drug_paths: list[KGPath],
    phenotype_coverage: int,
    node_names: dict[int, str],
    node_descriptions: dict[int, str] | None = None,
) -> str:
    """Serialize all evidence paths for a single drug.

    Args:
        drug_idx: Drug node index.
        drug_paths: All pruned paths ending at this drug.
        phenotype_coverage: Number of patient phenotypes this drug covers.
        node_names: Node index -> name mapping.
        node_descriptions: Optional node index -> description mapping.

    Returns:
        Structured text block for this drug candidate.
    """
    drug_name = node_names.get(drug_idx, str(drug_idx))
    lines = [f"Drug: {drug_name} (covers {phenotype_coverage} patient phenotypes)"]

    direct = [p for p in drug_paths if p.path_type == "direct_target"]
    mediated = [p for p in drug_paths if p.path_type == "pathway_mediated"]

    if direct:
        lines.append(f"  Direct target paths ({len(direct)}):")
        for p in direct:
            lines.append(f"    - {serialize_path(p, node_names, node_descriptions)}")

    if mediated:
        lines.append(f"  Pathway-mediated paths ({len(mediated)}):")
        for p in mediated:
            lines.append(f"    - {serialize_path(p, node_names, node_descriptions)}")

    return "\n".join(lines)


def build_prompt_context(
    candidate_drugs: dict[int, list[KGPath]],
    phenotype_coverage: dict[int, int],
    phenotype_names: list[str],
    node_names: dict[int, str],
    node_descriptions: dict[int, str] | None = None,
) -> str:
    """Build the full context block for the CoT LLM ranking prompt.

    Args:
        candidate_drugs: Mapping from drug_idx -> list of pruned paths.
        phenotype_coverage: Drug_idx -> phenotype coverage count.
        phenotype_names: List of patient phenotype names.
        node_names: Node index -> name mapping.
        node_descriptions: Optional node descriptions.

    Returns:
        Full prompt context string.
    """
    sections = [
        f"Patient phenotypes: {', '.join(phenotype_names)}",
        f"\nCandidate drugs ({len(candidate_drugs)} found via knowledge graph paths):\n",
    ]

    # Sort by coverage count descending
    sorted_drugs = sorted(
        candidate_drugs.items(),
        key=lambda x: phenotype_coverage.get(x[0], 0),
        reverse=True,
    )

    for drug_idx, paths in sorted_drugs:
        coverage = phenotype_coverage.get(drug_idx, 0)
        sections.append(serialize_drug_evidence(
            drug_idx, paths, coverage, node_names, node_descriptions,
        ))
        sections.append("")  # blank line separator

    return "\n".join(sections)
