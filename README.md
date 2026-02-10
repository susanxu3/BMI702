# BMI702 - Rare Disease AI Scientist

Given a patient's rare genetic variants (VCF), the system traces a path from **variant → affected cell type → disrupted gene/pathway → candidate drug**, producing ranked therapeutic hypotheses with mechanistic explanations. Most rare disease patients face a long diagnostic odyssey and limited treatment options. This project aims to accelerate the search for repurposable drugs by combining genomic analysis with AI reasoning.

## Workflow

The pipeline has four layers:

1. **Variant Annotation** — Scores rare variants for predicted pathogenicity using tools like CADD, AlphaMissense, and SpliceAI.
2. **Cell-Type & Gene Mapping** — Maps variants to the cell types and genes they likely affect using single-cell atlases and deep learning models.
3. **Knowledge Graph Integration** — Places disrupted genes and pathways onto a biomedical knowledge graph linking diseases, targets, drugs, and safety data.
4. **Drug Scoring & Reasoning** — Ranks candidate drugs using graph-based ML and generates mechanistic rationales via an LLM agent, including safety and contraindication checks.

![Pipeline Overview](Variant_to_Drug_Pipeline_Flowchart.pdf)

## Output

For each candidate drug, the system produces:

- A full **variant → enhancer → gene → pathway → drug** evidence chain
- **Cell-type context** for the predicted mechanism
- **Safety and contraindication** flags
- **Literature support** from PubMed and clinical trial databases
