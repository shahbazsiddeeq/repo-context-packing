# REPO CONTEXT PACKING

## Goal
Study whether structure-aware chunking and dependency-aware context packing improve repository-level code completion under fixed token budgets.

## Main pipeline
Query -> Retrieval -> Packing -> Model -> Evaluation

## Baseline
Fixed-size chunking + BM25 + top-k packing

## Proposed method
Structure-aware chunking + dependency-aware packing

## Datasets
- CodeSearchNet
- RepoEval

## Current status
- [ ] Repo setup
- [ ] Dataset preparation
- [ ] Baseline
- [ ] Smoke tests
- [ ] Proposed method
- [ ] Main experiments


# Final Paper Package

This package contains a cleaned paper draft and the core section/table files for your repository-level retrieval paper.

## Main result
On the full 1600-example RepoEval line-level split:
- Fixed BM25: MRR 0.7060, Recall@1 0.5831, Recall@5 0.8556
- Fixed BM25 + locality reranking: MRR 0.7394, Recall@1 0.6206, Recall@5 0.8831

## Recommended file to work from
Start from:
- `paper/full_draft_v2.md`

Then use:
- `paper/table1_main_result.md`
- `paper/result_highlights_v1.md`

for polishing the Results section and tables.
