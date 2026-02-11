# BMI702 - Rare Disease AI Scientist

Given a rare disease, the AI scientist produces ranked therapeutic hypotheses by tracing a path from variant → affected cell type → disrupted gene/pathway → candidate drug. 

## Workflow

The pipeline has the following layers:

1. **Common Variant GWAS integration**
2. **Rare Variant Association Testing** — Aggregates rare variants by gene or region and tests for disease association using burden tests, SKAT, and SKAT-O via frameworks like REGENIE. 
3. **Cell-Type & Gene Mapping** — Maps variants to the cell types and genes they likely affect using single-cell atlases and deep learning models.
4. **Knowledge Graph Integration** — Places candidate genes and pathways onto a biomedical knowledge graph, forming gene-disease-drug links.
5. **Drug Scoring & Reasoning** — Ranks candidate drugs using graph-based ML and generates mechanistic rationales via an LLM agent, including safety and contraindication checks.

## Output

For a given patient or rare disease, the system produces:

- Candidate drugs for repurposing, scored by mechanistic relevance
- Evidence tracing each from variant → gene → pathway → drug
- Cell-type context
- Safety flags
- Literature support
