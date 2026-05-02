"""Phenotype-anchored subgraph extraction (Section 5.2.2 of midterm report).

Anchored to patient HPO phenotype nodes (NOT disease nodes).
Traverses two disease-free path types in parallel across PrimeKG:

Path Type 1 — Direct Target:
  phenotype → protein → drug
  via gene_phenotype and drug_protein edges.
  Captures drugs that directly modulate phenotype-associated proteins.

Path Type 2 — Pathway-Mediated:
  phenotype → protein → pathway → drug
  via protein_pathway and drug_pathway Reactome edges.
  Captures drugs acting on same biological pathway without a direct
  protein-drug edge.

The union of both path types forms the candidate drug pool D.
Runs on CPU to avoid GPU memory pressure during LLM inference.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class KGPath:
    """A single path from a phenotype to a drug through the KG."""
    path_type: str                    # "direct_target" or "pathway_mediated"
    phenotype_idx: int
    protein_idx: int
    drug_idx: int
    pathway_idx: int | None = None    # only for pathway-mediated paths
    edge_types: list[str] = field(default_factory=list)

    @property
    def intermediate_nodes(self) -> list[int]:
        nodes = [self.protein_idx]
        if self.pathway_idx is not None:
            nodes.append(self.pathway_idx)
        return nodes


def build_adjacency_maps(
    kg: pd.DataFrame, src_col: str = "x_index", dst_col: str = "y_index", rel_col: str = "relation"
) -> dict[str, dict[int, set[int]]]:
    """Build per-relation adjacency maps for path traversal.

    Args:
        kg: PrimeKG DataFrame.
        src_col: Source node column.
        dst_col: Destination node column.
        rel_col: Relation type column.

    Returns:
        Dict mapping relation name -> {src_node: {dst_nodes}}.
    """
    adj: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for _, row in kg.iterrows():
        rel = row[rel_col]
        adj[rel][row[src_col]].add(row[dst_col])
        adj[rel][row[dst_col]].add(row[src_col])  # undirected traversal
    return dict(adj)


def extract_direct_target_paths(
    phenotype_indices: list[int],
    adj: dict[str, dict[int, set[int]]],
) -> list[KGPath]:
    """Extract Path Type 1: phenotype → protein → drug.

    Args:
        phenotype_indices: Patient's HPO phenotype node indices.
        adj: Per-relation adjacency maps from build_adjacency_maps.

    Returns:
        List of KGPath objects for direct target paths.
    """
    gene_pheno = adj.get("gene_phenotype", {})
    drug_protein = adj.get("drug_protein", {})

    paths = []
    for pheno in phenotype_indices:
        proteins = gene_pheno.get(pheno, set())
        for protein in proteins:
            drugs = drug_protein.get(protein, set())
            for drug in drugs:
                paths.append(KGPath(
                    path_type="direct_target",
                    phenotype_idx=pheno,
                    protein_idx=protein,
                    drug_idx=drug,
                    edge_types=["gene_phenotype", "drug_protein"],
                ))
    return paths


def extract_pathway_mediated_paths(
    phenotype_indices: list[int],
    adj: dict[str, dict[int, set[int]]],
) -> list[KGPath]:
    """Extract Path Type 2: phenotype → protein → pathway → drug.

    Args:
        phenotype_indices: Patient's HPO phenotype node indices.
        adj: Per-relation adjacency maps from build_adjacency_maps.

    Returns:
        List of KGPath objects for pathway-mediated paths.
    """
    gene_pheno = adj.get("gene_phenotype", {})
    prot_pathway = adj.get("protein_pathway", {})
    drug_pathway = adj.get("drug_pathway", {})

    paths = []
    for pheno in phenotype_indices:
        proteins = gene_pheno.get(pheno, set())
        for protein in proteins:
            pathways = prot_pathway.get(protein, set())
            for pathway in pathways:
                drugs = drug_pathway.get(pathway, set())
                for drug in drugs:
                    paths.append(KGPath(
                        path_type="pathway_mediated",
                        phenotype_idx=pheno,
                        protein_idx=protein,
                        drug_idx=drug,
                        pathway_idx=pathway,
                        edge_types=["gene_phenotype", "protein_pathway", "drug_pathway"],
                    ))
    return paths


def extract_all_paths(
    phenotype_indices: list[int],
    adj: dict[str, dict[int, set[int]]],
) -> tuple[list[KGPath], set[int]]:
    """Extract paths via both path types and return candidate drug pool D.

    Args:
        phenotype_indices: Patient's HPO phenotype node indices.
        adj: Per-relation adjacency maps.

    Returns:
        all_paths: Union of direct target and pathway-mediated paths.
        candidate_drugs: Set of drug indices reachable via any path (pool D).
    """
    direct = extract_direct_target_paths(phenotype_indices, adj)
    mediated = extract_pathway_mediated_paths(phenotype_indices, adj)
    all_paths = direct + mediated
    candidate_drugs = {p.drug_idx for p in all_paths}
    return all_paths, candidate_drugs
