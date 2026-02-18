# Phenotype-Conditioned Drug Prediction for Rare Diseases

Given a rare disease patient's phenotype set, the model produces ranked drug candidates by dynamically constructing a patient-specific disease representation via drug-conditioned cross-attention over phenotype embeddings on a biomedical knowledge graph. The insiprations are:
- **TxGNN** (Huang et al., 2024) — zero-shot drug repurposing on biomedical knowledge graphs; we extend its graph-based drug prediction framework to handle patient phenotype sets rather than fixed disease nodes
- **SHEPHERD** (Alsentzer et al., 2023) — GNN-based rare disease diagnosis from phenotype and variant inputs; we adopt the phenotype-as-input paradigm but redirect it toward drug prediction with drug-conditioned aggregation
- **PINNACLE** (Arakelyan et al., 2024) — context-conditioned protein embeddings; we apply the same conditioning intuition to disease representations, making them drug-conditioned rather than cell-type-conditioned

## Workflow

1. **Knowledge Graph Encoding** — R-GCN over a graph of genes, drugs, phenotypes, and pathways (Open Targets, HPO, STRING). Each node gets a learned embedding.
2. **Phenotype Set Encoding** — Each candidate drug embedding acts as the query; patient HPO phenotype embeddings act as keys/values. The model attends to different phenotypes when scoring different drugs.
3. **Drug Scoring** — Cross-attention output (a drug-conditioned disease representation) is scored against each drug embedding to produce a ranked list.

## Data

Open Targets — gene-disease and drug-disease associations

Human Phenotype Ontology (HPO) — phenotype terms and hierarchy

STRING — protein-protein interaction network

GWAS Catalog — common variant-to-phenotype associations used as additional graph edges (Optional)

## Model
A two-component model trained end-to-end:

R-GCN — trained on the biomedical knowledge graph to produce node embeddings for genes, drugs, phenotypes, and pathways

Cross-attention layer — trained to produce drug-conditioned disease representations from patient HPO phenotype sets; drug embeddings serve as queries, phenotype embeddings as keys and values

## Output

For a given patient phenotype set, the system produces:
- Ranked candidate drugs for repurposing, scored by mechanistic relevance
- Attention weights tracing each recommendation to the phenotypes that drove it
- Interpretable rationale (e.g., *"losartan ← aortic root dilation 0.6, mitral valve prolapse 0.2"*)
- Zero-shot predictions for undiagnosed patients with arbitrary phenotype combinations

## Key Innovation

Unlike TxGNN, which represents diseases as fixed graph nodes, this model treats a patient as a **set of HPO phenotypes** and dynamically constructs a disease representation at inference time — enabling drug prediction without a confirmed diagnosis.

## Implementation

- Standard R-GCN (PyTorch Geometric) + transformer cross-attention (PyTorch). 
- Graph constructed via Open Targets API. 
- Runs on Google Colab.
