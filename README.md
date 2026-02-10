# BMI702 - Rare Disease AI Scientist

Given a patient's rare genetic variants (VCF), the system traces a path from **variant → affected cell type → disrupted gene/pathway → candidate drug**, producing ranked therapeutic hypotheses with mechanistic explanations. Most rare disease patients face a long diagnostic odyssey and limited treatment options. This project aims to accelerate the search for repurposable drugs by combining genomic analysis with AI reasoning.

## Workflow

The pipeline has four layers:

1. **Variant Annotation** — Scores rare variants for predicted pathogenicity using tools like CADD.
2. **Rare Variant Association Testing** — Aggregates rare variants by gene or region and tests for disease association using burden tests, SKAT, and SKAT-O via frameworks like REGENIE. 
3. **Cell-Type & Gene Mapping** — Maps variants to the cell types and genes they likely affect using single-cell atlases and deep learning models.
4. **Knowledge Graph Integration** — Places candidate genes and pathways onto a biomedical knowledge graph, forming gene-disease-drug links.
5. **Drug Scoring & Reasoning** — Ranks candidate drugs using graph-based ML and generates mechanistic rationales via an LLM agent, including safety and contraindication checks.

## Output

For a given patient or rare disease, the system produces:

- Ranked list of candidate drugs for repurposing, scored by mechanistic relevance
- Evidence chains tracing each hypothesis from variant → gene → pathway → drug
- Cell-type context identifying which cell types are likely affected
- Safety and contraindication flags
- Literature support
