# Phenotype-Conditioned Drug Prediction for Rare Diseases

Given a rare disease patient's phenotype set, the model produces ranked drug candidates by dynamically constructing a patient-specific disease representation via **drug-conditioned cross-attention** over phenotype embeddings on a biomedical knowledge graph.

## Workflow

1. **Knowledge Graph Encoding** — R-GCN over a graph of genes, drugs, phenotypes, and pathways (Open Targets, HPO, STRING). Each node gets a learned embedding.
2. **Phenotype Set Encoding** — Each candidate drug embedding acts as the query; patient HPO phenotype embeddings act as keys/values. The model attends to different phenotypes when scoring different drugs.
3. **Drug Scoring** — Cross-attention output (a drug-conditioned disease representation) is scored against each drug embedding to produce a ranked list.

## Output

For a given patient phenotype set, the system produces:
- Ranked candidate drugs for repurposing, scored by mechanistic relevance
- Attention weights tracing each recommendation to the phenotypes that drove it
- Interpretable rationale (e.g., *"losartan ← aortic root dilation 0.6, mitral valve prolapse 0.2"*)
- Zero-shot predictions for undiagnosed patients with arbitrary phenotype combinations

## Key Design Choice

Unlike TxGNN, which represents diseases as fixed graph nodes, this model treats a patient as a **set of HPO phenotypes** and dynamically constructs a disease representation at inference time — enabling drug prediction without a confirmed diagnosis.

## Implementation

Standard R-GCN (PyTorch Geometric) + transformer cross-attention (PyTorch). Graph constructed via Open Targets API. Runs on Google Colab.
