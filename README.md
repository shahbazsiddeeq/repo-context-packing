# Repo Context Packing

Research project on locality-aware retrieval for repository-level code completion.

**Core finding:** A simple locality-aware reranking method (same-file bonus + line-distance penalty on top of BM25) is the strongest context-construction strategy tested on RepoEval line-level completion, outperforming structure-aware chunking, hybrid chunking, and packing heuristics.

---

## Pipeline

```
Repo files
  → Chunking  (fixed-size | structure-aware | hybrid)
  → BM25 Retrieval  (top-k=10)
  → Locality-Aware Reranking  [main contribution]
  → Context Packing
  → LM Generation  (StarCoder2-7B)
  → Evaluation  (MRR / Recall@k / Exact Match)
```

---

## Key Results (RepoEval, 1600 examples)

### Retrieval

| Method | MRR | Recall@1 | Recall@5 |
|---|---|---|---|
| Fixed BM25 | 0.7060 | 0.5831 | 0.8556 |
| Fixed BM25 + Locality Reranking | **0.7394** | **0.6206** | **0.8831** |

### End-to-End Completion (StarCoder2-7B, Exact Match)

| Condition | EM |
|---|---|
| Prompt-only | 34.62% |
| BM25 top-1 context | 78.19% |
| Locality top-1 context | **79.38%** |

All gains are statistically significant (McNemar p < 0.005 for locality vs BM25).

---

## Project Structure

```
repo-context-packing/
├── src/
│   ├── chunking/
│   │   ├── fixed_chunker.py        # overlapping fixed-size windows
│   │   ├── structure_chunker.py    # AST-based (Python, tree-sitter)
│   │   └── hybrid_chunker.py       # structure units merged into fixed windows
│   ├── retrieval/
│   │   ├── bm25_retriever.py       # BM25Okapi over identifier-tokenized chunks
│   │   └── locality_reranker.py    # same-file bonus + line-distance penalty
│   ├── packing/
│   │   ├── topk_packer.py          # top-k chunks within token budget
│   │   ├── neighbor_packer.py      # locality-aware neighbor selection
│   │   ├── fixed_neighbor_packer.py
│   │   └── dependency_packer.py    # import/dependency-aware packing
│   ├── evaluation/
│   │   ├── eval_retrieval.py       # MRR, Recall@1, Recall@5
│   │   └── eval_downstream.py      # exact match for completion
│   ├── data/                       # data loaders
│   └── utils/
├── scripts/
│   ├── run_repoeval_completion_step1.py   # main end-to-end completion run
│   ├── run_repoeval_full_locality.py      # full 1600-example retrieval run
│   ├── run_repoeval_locality_comparison.py
│   ├── run_repoeval_packing_comparison.py
│   ├── sweep_locality_reranker.py         # hyperparameter sweep (beta, lambda)
│   ├── sweep_structure_chunk_size.py
│   ├── run_crosscodeeval_official_prompts.py
│   ├── analyze_full_locality_results.py
│   └── make_paper_tables.py
├── configs/
│   ├── base.yaml
│   ├── baseline.yaml
│   └── method.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
├── outputs/
│   ├── completion_step1_1600/      # full end-to-end results
│   ├── full_run_updated_good_results/
│   ├── tables/
│   ├── figures/
│   └── logs/
├── paper/                          # all paper drafts and tables
│   ├── paperreadme.md              # paper-specific README
│   ├── full_draft_v2.md            # current full paper draft
│   ├── research_status_and_plan.txt
│   └── ...
├── cceval/                         # CrossCodeEval setup
├── tests/
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `pyyaml`, `tqdm`, `rank-bm25`, `scikit-learn`, `matplotlib`, `tree-sitter`, `pytest`, `torch`, `transformers`.

---

## Running Experiments

### Full RepoEval retrieval + locality reranking

```bash
python scripts/run_repoeval_full_locality.py
```

### End-to-end completion (StarCoder2-7B, requires GPU)

```bash
python scripts/run_repoeval_completion_step1.py
```

Key settings in `run_repoeval_completion_step1.py`:
- `CHUNK_SIZE = 120`, `CHUNK_OVERLAP = 20`
- `SAME_FILE_BONUS = 150.0`, `LINE_DISTANCE_WEIGHT = 0.2`
- `MODEL_NAME = "bigcode/starcoder2-7b"`

### Hyperparameter sweep (locality reranker)

```bash
python scripts/sweep_locality_reranker.py
```

### Generate paper tables

```bash
python scripts/make_paper_tables.py
python scripts/make_more_paper_tables.py
```

---

## Method: Locality-Aware Reranking

For each BM25-retrieved chunk, the reranked score is:

```
s'(c) = s_BM25(c) + β · I[same-file] − λ · d(c, l)
```

- `β = 150.0` — same-file bonus
- `λ = 0.2` — line-distance penalty weight
- `d(c, l)` — line distance between chunk span and target completion line

Implementation: `src/retrieval/locality_reranker.py`

---

## Benchmarks

| Benchmark | Status |
|---|---|
| RepoEval (1600 examples, line-level) | Complete |
| CrossCodeEval Python (smoke test) | Generation done, scoring in progress |
| CrossCodeEval Java / TypeScript / C# | Not started |

---

## Paper

See [`paper/paperreadme.md`](paper/paperreadme.md) for full paper documentation including research questions, all result tables, statistical tests, and submission roadmap.

Current draft: [`paper/full_draft_v2.md`](paper/full_draft_v2.md) (raw, not finalized).

**Target venue:** Strong SE or AI conference (ICSE, ASE, EMNLP). Journal submission after CrossCodeEval and additional model validation.
