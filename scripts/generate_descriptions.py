"""Generate textual descriptions for all PrimeKG drugs and phenotypes.

Two stages:
  Stage A — PrimeKG metadata only (Tier 1: name, Tier 2: name + 1-hop KG context)
  Stage B — GPT-4 enrichment from Tier 2 metadata (optional, requires OpenAI API key)

Descriptions are used for:
  1. Text encoder input (cache_embeddings.py)
  2. GraphRAG path serialization (path_serializer.py)

Entities to describe:
  - 7,957 drugs
  - Phenotypes associated with the 539 eligible diseases

Output: data/descriptions/{drugs,phenotypes}_{tier1,tier2,gpt4o}.json
  Format: {"node_index": {"name": str, "text": str}, ...}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── PrimeKG constants (matching src/data/primekg_loader.py) ──────────────
SRC_COL = "x_index"
DST_COL = "y_index"
REL_COL = "relation"

DRUG_TYPE = "drug"
DISEASE_TYPE = "disease"
PHENOTYPE_TYPE = "effect/phenotype"
PROTEIN_TYPE = "gene/protein"
PATHWAY_TYPE = "pathway"

# Edge types relevant for drug descriptions
DRUG_PROTEIN_REL = "drug_protein"
DRUG_PATHWAY_REL = "drug_pathway"  # not in all PrimeKG versions; fallback via protein
INDICATION_REL = "indication"
OFF_LABEL_REL = "off-label use"
CONTRAINDICATION_REL = "contraindication"

# Edge types relevant for phenotype descriptions
GENE_PHENOTYPE_REL = "phenotype_protein"  # PrimeKG uses phenotype_protein
DISEASE_PHENOTYPE_REL = "disease_phenotype_positive"


# ── Helper: build node name lookup ───────────────────────────────────────
def _build_name_lookup(nodes_df: pd.DataFrame) -> dict[int, str]:
    """Map node_index -> node_name for all nodes."""
    return dict(zip(nodes_df["node_index"], nodes_df["node_name"]))


def _build_type_lookup(nodes_df: pd.DataFrame) -> dict[int, str]:
    """Map node_index -> node_type for all nodes."""
    return dict(zip(nodes_df["node_index"], nodes_df["node_type"]))


# ── Helper: build adjacency by relation ──────────────────────────────────
def _build_adjacency(
    kg_df: pd.DataFrame,
    relation: str,
    directed: bool = False,
) -> dict[int, set[int]]:
    """Build node -> neighbors map for a given relation type.

    Args:
        kg_df: Full KG DataFrame.
        relation: Relation name to filter on.
        directed: If False, treat edges as undirected (add both directions).

    Returns:
        Adjacency dict mapping node index to set of neighbor indices.
    """
    edges = kg_df[kg_df[REL_COL] == relation]
    adj: dict[int, set[int]] = defaultdict(set)
    for _, row in edges.iterrows():
        adj[row[SRC_COL]].add(row[DST_COL])
        if not directed:
            adj[row[DST_COL]].add(row[SRC_COL])
    return adj


# ── Stage A: PrimeKG metadata descriptions ───────────────────────────────
def build_drug_descriptions(
    nodes_df: pd.DataFrame,
    kg_df: pd.DataFrame,
    tier: str = "tier2",
    exclude_diseases: set[int] | None = None,
) -> dict[int, dict]:
    """Build text descriptions for all drug nodes.

    Args:
        nodes_df: PrimeKG nodes DataFrame.
        kg_df: PrimeKG kg DataFrame.
        tier: "tier1" (name only) or "tier2" (name + 1-hop KG context).
        exclude_diseases: Disease node indices to exclude from indication
            context (test diseases, for leakage prevention).

    Returns:
        Dict mapping node_index -> {"name": str, "text": str}.
    """
    if exclude_diseases is None:
        exclude_diseases = set()

    name_lookup = _build_name_lookup(nodes_df)
    type_lookup = _build_type_lookup(nodes_df)
    drug_nodes = nodes_df[nodes_df["node_type"] == DRUG_TYPE]

    descriptions: dict[int, dict] = {}

    if tier == "tier1":
        for _, row in drug_nodes.iterrows():
            idx = row["node_index"]
            name = row["node_name"]
            descriptions[idx] = {"name": name, "text": name}
        return descriptions

    # Tier 2: name + 1-hop KG context
    # Build adjacency maps for relevant relations
    drug_protein_adj = _build_adjacency(kg_df, DRUG_PROTEIN_REL)
    indication_adj = _build_adjacency(kg_df, INDICATION_REL)
    offlabel_adj = _build_adjacency(kg_df, OFF_LABEL_REL)

    # Check if drug_pathway relation exists
    has_drug_pathway = DRUG_PATHWAY_REL in kg_df[REL_COL].values
    if has_drug_pathway:
        drug_pathway_adj = _build_adjacency(kg_df, DRUG_PATHWAY_REL)
    else:
        # Fallback: drug -> protein -> pathway
        protein_pathway_adj = _build_adjacency(kg_df, "protein_pathway")
        drug_pathway_adj: dict[int, set[int]] = defaultdict(set)
        for drug_idx, proteins in drug_protein_adj.items():
            for prot in proteins:
                drug_pathway_adj[drug_idx].update(protein_pathway_adj.get(prot, set()))

    for _, row in drug_nodes.iterrows():
        idx = row["node_index"]
        name = row["node_name"]

        parts = [name + "."]

        # Targets (proteins)
        targets = drug_protein_adj.get(idx, set())
        target_names = sorted(
            name_lookup.get(t, str(t)) for t in targets
            if type_lookup.get(t) == PROTEIN_TYPE
        )
        if target_names:
            parts.append(f"Targets: {', '.join(target_names[:10])}.")

        # Pathways
        pathways = drug_pathway_adj.get(idx, set())
        pathway_names = sorted(
            name_lookup.get(p, str(p)) for p in pathways
            if type_lookup.get(p) == PATHWAY_TYPE
        )
        if pathway_names:
            parts.append(f"Pathways: {', '.join(pathway_names[:5])}.")

        # Indications (exclude test diseases)
        indications = indication_adj.get(idx, set()) - exclude_diseases
        offlabel = offlabel_adj.get(idx, set()) - exclude_diseases
        all_diseases = indications | offlabel
        disease_names = sorted(
            name_lookup.get(d, str(d)) for d in all_diseases
            if type_lookup.get(d) == DISEASE_TYPE
        )
        if disease_names:
            parts.append(f"Indications: {', '.join(disease_names[:10])}.")

        descriptions[idx] = {"name": name, "text": " ".join(parts)}

    return descriptions


def build_phenotype_descriptions(
    nodes_df: pd.DataFrame,
    kg_df: pd.DataFrame,
    tier: str = "tier2",
    exclude_nodes: set[int] | None = None,
    phenotype_indices: set[int] | None = None,
) -> dict[int, dict]:
    """Build text descriptions for phenotype nodes.

    Args:
        nodes_df: PrimeKG nodes DataFrame.
        kg_df: PrimeKG kg DataFrame.
        tier: "tier1" (name only) or "tier2" (name + 1-hop KG context).
        exclude_nodes: Node indices to exclude from 1-hop context
            (test disease nodes, for leakage prevention).
        phenotype_indices: If provided, only describe these phenotype nodes.
            If None, describe all phenotype-type nodes.

    Returns:
        Dict mapping node_index -> {"name": str, "text": str}.
    """
    if exclude_nodes is None:
        exclude_nodes = set()

    name_lookup = _build_name_lookup(nodes_df)
    type_lookup = _build_type_lookup(nodes_df)

    if phenotype_indices is not None:
        pheno_nodes = nodes_df[
            (nodes_df["node_type"] == PHENOTYPE_TYPE)
            & (nodes_df["node_index"].isin(phenotype_indices))
        ]
    else:
        pheno_nodes = nodes_df[nodes_df["node_type"] == PHENOTYPE_TYPE]

    descriptions: dict[int, dict] = {}

    if tier == "tier1":
        for _, row in pheno_nodes.iterrows():
            idx = row["node_index"]
            name = row["node_name"]
            descriptions[idx] = {"name": name, "text": name}
        return descriptions

    # Tier 2: name + 1-hop KG context
    gene_pheno_adj = _build_adjacency(kg_df, GENE_PHENOTYPE_REL)
    disease_pheno_adj = _build_adjacency(kg_df, DISEASE_PHENOTYPE_REL)

    for _, row in pheno_nodes.iterrows():
        idx = row["node_index"]
        name = row["node_name"]

        parts = [name + "."]

        # Associated genes/proteins
        genes = gene_pheno_adj.get(idx, set())
        gene_names = sorted(
            name_lookup.get(g, str(g)) for g in genes
            if type_lookup.get(g) == PROTEIN_TYPE
        )
        if gene_names:
            parts.append(f"Associated genes: {', '.join(gene_names[:10])}.")

        # Related diseases (excluding test diseases)
        diseases = disease_pheno_adj.get(idx, set()) - exclude_nodes
        disease_names = sorted(
            name_lookup.get(d, str(d)) for d in diseases
            if type_lookup.get(d) == DISEASE_TYPE
        )
        if disease_names:
            parts.append(f"Associated conditions: {', '.join(disease_names[:10])}.")

        descriptions[idx] = {"name": name, "text": " ".join(parts)}

    return descriptions


# ── Stage B: GPT-4 enrichment ────────────────────────────────────────────

DRUG_PROMPT = """The following is structured information about a drug from a biomedical knowledge graph:
{tier2_text}

Using this as context to identify the drug, write a concise 2-3 sentence scientific description that captures its mechanism of action, therapeutic class, known clinical applications, and any emerging or investigational uses. Draw on your full biomedical knowledge beyond what is listed above."""

PHENOTYPE_PROMPT = """The following is structured information about a clinical phenotype from a biomedical knowledge graph:
{tier2_text}

Using this as context to identify the phenotype, write a concise 2-3 sentence description covering its molecular basis, associated biological pathways, and clinical presentation. Draw on your full biomedical knowledge beyond what is listed above. Do not name specific diseases."""


def enrich_with_llm(
    descriptions: dict[int, dict],
    entity_type: str,
    model: str = "gpt-4o",
    batch_size: int = 50,
    output_path: Path | None = None,
    max_retries: int = 3,
) -> dict[int, dict]:
    """Enrich Tier 2 descriptions using GPT-4.

    Uses the KG metadata as a grounding anchor for entity disambiguation,
    then invites GPT-4 to draw on its full parametric biomedical knowledge.

    Args:
        descriptions: Tier 2 descriptions {node_index: {"name": str, "text": str}}.
        entity_type: "drug" or "phenotype" — selects the prompt template.
        model: OpenAI model name.
        batch_size: Number of entities per API batch.
        output_path: If provided, saves intermediate results for resumability.
        max_retries: Max retries per failed API call.

    Returns:
        Enriched descriptions with "llm_text" key added.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai  — required for GPT-4 enrichment")

    client = OpenAI()
    prompt_template = DRUG_PROMPT if entity_type == "drug" else PHENOTYPE_PROMPT

    # Load existing progress if resuming
    enriched = {}
    if output_path and output_path.exists():
        with open(output_path) as f:
            enriched = json.load(f)
        logger.info(f"Resuming from {len(enriched)} existing enrichments")

    remaining = {
        k: v for k, v in descriptions.items() if str(k) not in enriched
    }
    items = list(remaining.items())
    logger.info(f"Enriching {len(items)} {entity_type}s with {model}")

    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start : batch_start + batch_size]

        for node_idx, desc in batch:
            prompt = prompt_template.format(tier2_text=desc["text"])

            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=200,
                    )
                    llm_text = response.choices[0].message.content.strip()
                    enriched[str(node_idx)] = {
                        "name": desc["name"],
                        "text": llm_text,
                    }
                    break
                except Exception as e:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for "
                        f"{entity_type} {node_idx}: {e}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        logger.error(f"Skipping {entity_type} {node_idx} after {max_retries} failures")
                        enriched[str(node_idx)] = {
                            "name": desc["name"],
                            "text": desc["text"],  # fallback to Tier 2
                        }

        # Save intermediate results
        if output_path:
            with open(output_path, "w") as f:
                json.dump(enriched, f, indent=2)
            logger.info(
                f"Saved {len(enriched)} enrichments "
                f"({batch_start + len(batch)}/{len(items)} done)"
            )

    return enriched


# ── Collect phenotype indices from eligible diseases ─────────────────────
def get_disease_phenotype_indices(
    kg_df: pd.DataFrame,
    disease_indices: set[int],
) -> set[int]:
    """Get all phenotype node indices associated with a set of diseases.

    Args:
        kg_df: PrimeKG kg DataFrame.
        disease_indices: Set of disease node indices.

    Returns:
        Set of phenotype node indices.
    """
    pheno_edges = kg_df[kg_df[REL_COL] == DISEASE_PHENOTYPE_REL]
    pheno_indices: set[int] = set()

    for _, row in pheno_edges.iterrows():
        src, dst = row[SRC_COL], row[DST_COL]
        if src in disease_indices:
            pheno_indices.add(dst)
        if dst in disease_indices:
            pheno_indices.add(src)

    return pheno_indices


# ── Main ─────────────────────────────────────────────────────────────────
def main(
    data_dir: str,
    split_dir: str,
    output_dir: str,
    tier: str = "tier2",
    use_llm: bool = False,
    llm_model: str = "gpt-4o",
) -> None:
    """Generate descriptions for drugs and phenotypes.

    Args:
        data_dir: Path to PrimeKG data directory (nodes.csv, kg.csv).
        split_dir: Path to train/test split files.
        output_dir: Output directory for JSON files.
        tier: "tier1" or "tier2" for metadata descriptions.
        use_llm: If True, enrich Tier 2 with GPT-4.
        llm_model: OpenAI model for enrichment.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    data_dir = Path(data_dir)
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load PrimeKG
    logger.info("Loading PrimeKG...")
    print(os.listdir(data_dir))
    nodes_df = pd.read_csv(data_dir / "nodes.csv")
    kg_df = pd.read_csv(data_dir / "kg.csv")
    logger.info(f"Loaded {len(nodes_df)} nodes, {len(kg_df)} edges")

    # Load split for leakage prevention
    test_diseases = set(
        int(x.strip())
        for x in (split_dir / "test_disease_ids.txt").read_text().splitlines()
        if x.strip()
    )
    train_diseases = set(
        int(x.strip())
        for x in (split_dir / "train_disease_ids.txt").read_text().splitlines()
        if x.strip()
    )
    all_diseases = train_diseases | test_diseases
    logger.info(
        f"Split: {len(train_diseases)} train, {len(test_diseases)} test diseases"
    )

    # Collect phenotype indices from all eligible diseases
    pheno_indices = get_disease_phenotype_indices(kg_df, all_diseases)
    logger.info(f"Phenotypes associated with eligible diseases: {len(pheno_indices)}")

    # Build descriptions
    logger.info(f"Building {tier} drug descriptions...")
    drug_descs = build_drug_descriptions(
        nodes_df, kg_df, tier=tier, exclude_diseases=test_diseases
    )
    logger.info(f"Built {len(drug_descs)} drug descriptions")

    logger.info(f"Building {tier} phenotype descriptions...")
    pheno_descs = build_phenotype_descriptions(
        nodes_df, kg_df, tier=tier,
        exclude_nodes=test_diseases,
        phenotype_indices=pheno_indices,
    )
    logger.info(f"Built {len(pheno_descs)} phenotype descriptions")

    if use_llm:
        # Enrich with GPT-4
        suffix = llm_model.replace("-", "").replace(".", "")
        logger.info(f"Enriching drugs with {llm_model}...")
        drug_descs = enrich_with_llm(
            drug_descs,
            entity_type="drug",
            model=llm_model,
            output_path=output_dir / f"drugs_{suffix}.json",
        )
        logger.info(f"Enriching phenotypes with {llm_model}...")
        pheno_descs = enrich_with_llm(
            pheno_descs,
            entity_type="phenotype",
            model=llm_model,
            output_path=output_dir / f"phenotypes_{suffix}.json",
        )
        drug_path = output_dir / f"drugs_{suffix}.json"
        pheno_path = output_dir / f"phenotypes_{suffix}.json"
    else:
        drug_path = output_dir / f"drugs_{tier}.json"
        pheno_path = output_dir / f"phenotypes_{tier}.json"

    # Save
    with open(drug_path, "w") as f:
        json.dump(
            {str(k): v for k, v in drug_descs.items()},
            f,
            indent=2,
        )
    with open(pheno_path, "w") as f:
        json.dump(
            {str(k): v for k, v in pheno_descs.items()},
            f,
            indent=2,
        )

    logger.info(f"Saved: {drug_path} ({len(drug_descs)} drugs)")
    logger.info(f"Saved: {pheno_path} ({len(pheno_descs)} phenotypes)")

    # Print sample descriptions
    sample_drug = next(iter(drug_descs.values()))
    sample_pheno = next(iter(pheno_descs.values()))
    logger.info(f"Sample drug description: {sample_drug['text'][:200]}")
    logger.info(f"Sample phenotype description: {sample_pheno['text'][:200]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate entity descriptions")
    parser.add_argument(
        "--data-dir", type=str, default="data/primekg",
        help="Path to PrimeKG data directory",
    )
    parser.add_argument(
        "--split-dir", type=str, default="data/splits",
        help="Path to train/test split files",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/descriptions",
        help="Output directory for description JSON files",
    )
    parser.add_argument(
        "--tier", type=str, default="tier2", choices=["tier1", "tier2"],
        help="Description tier: tier1 (name only) or tier2 (name + KG context)",
    )
    parser.add_argument(
        "--use-llm", action="store_true",
        help="Enrich Tier 2 descriptions with GPT-4",
    )
    parser.add_argument(
        "--llm-model", type=str, default="gpt-4o",
        help="OpenAI model for LLM enrichment",
    )
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        tier=args.tier,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
    )
