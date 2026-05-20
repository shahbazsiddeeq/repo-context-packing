# Locality-Aware Retrieval for Repository-Level Code Completion

Replication package for an empirical study on locality-aware reranking for repository-level line-level code completion.

## Method

For each BM25-retrieved chunk, the reranked score is:

```
s'(c) = s_BM25(c) + β · I[same-file] − λ · d(c, l)
```

- `β = 150.0` — same-file bonus
- `λ = 0.2` — line-distance penalty weight
- `d(c, l)` — minimum line distance from chunk span to target completion line

Implementation: `src/retrieval/locality_reranker.py`

---

## Key Results

### Retrieval (RepoEval, 1,600 instances)

| Method | MRR | Recall@1 | Recall@5 |
|---|---|---|---|
| BM25 (fixed-size chunking) | 0.7060 | 0.5831 | 0.8556 |
| BM25 + Locality Reranking | **0.7394** | **0.6206** | **0.8831** |
| Dense (UniXcoder) | 0.6183 | 0.4863 | 0.7875 |
| Dense + Locality Reranking | **0.7719** | **0.7338** | 0.8225 |

### End-to-End Completion (Exact Match, RepoEval 1,600 instances)

| Model | BM25 | Locality Reranking | Δ |
|---|---|---|---|
| StarCoder2-7B | 78.19% | 79.38% | +1.19 pp |
| StarCoder2-3B | 77.62% | 79.19% | +1.56 pp |
| CodeLlama-7B | 76.06% | 77.56% | +1.50 pp |
| DeepSeek-Coder-6.7B | 76.62% | 78.38% | +1.75 pp |
| CodeQwen1.5-7B | 78.12% | 79.50% | +1.38 pp |

All gains are statistically significant (McNemar p < 0.005).

Dense + Locality Reranking raises end-to-end exact match from 73.88% to 90.06% (+16.19 pp, McNemar p = 2.49×10⁻⁶⁸).

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `pyyaml`, `tqdm`, `rank-bm25`, `scikit-learn`, `matplotlib`, `tree-sitter`, `torch`, `transformers`, `faiss-cpu`.

---

## Reproducing Results

### Retrieval evaluation (BM25 + locality reranking)

```bash
python scripts/run_repoeval_full_locality.py
```

### End-to-end completion (requires GPU, ~24 GB VRAM for 7B models)

```bash
python scripts/run_repoeval_completion_model.py \
  --model_name bigcode/starcoder2-7b \
  --output_dir outputs/my_run
```

### Dense retrieval (UniXcoder + FAISS)

```bash
python scripts/run_dense_retrieval.py
python scripts/run_dense_completion_model.py
```

### Hyperparameter sweep (β × λ grid, 49 combinations)

```bash
python scripts/sweep_locality_reranker.py
```

### Chunking comparison (fixed-size vs structure-aware vs hybrid)

```bash
python scripts/run_chunking_comparison_full.py
```

### CrossCodeEval

```bash
python scripts/run_crosscodeeval_official_prompts.py
```

---

## Data

Experiments use [RepoEval](https://github.com/microsoft/RepoEval) (1,600 Python line-level completion instances from 8 open-source repositories). Download instructions: `scripts/data_prep/`.

---

## Project Structure

```
src/
├── chunking/       fixed-size, structure-aware (AST/tree-sitter), hybrid
├── retrieval/      BM25 retriever + locality reranker
├── packing/        top-k, neighbor, dependency-aware context packing
├── evaluation/     retrieval (MRR, Recall@k) and completion (exact match)
└── data/           RepoEval and CodeSearchNet loaders

scripts/            experiment runners and analysis scripts
outputs/            pre-computed results (CSV + JSON + TXT)
configs/            YAML experiment configurations
docs/               pilot results
```

---

## Pre-computed Outputs

Results are in `outputs/` and can be used to verify numbers without re-running:

| Directory | Contents |
|---|---|
| `outputs/completion_step1_1600/` | StarCoder2-7B BM25 results |
| `outputs/completion_starcoder2_3b_1600/` | StarCoder2-3B results |
| `outputs/completion_codellama_7b_1600/` | CodeLlama-7B results |
| `outputs/completion_deepseek_6b_1600/` | DeepSeek-Coder-6.7B results |
| `outputs/completion_codeqwen_7b_1600/` | CodeQwen-7B results |
| `outputs/completion_dense_starcoder2_7b/` | Dense + Locality end-to-end results |
| `outputs/chunking_comparison_1600/` | Chunking strategy comparison (1,600 instances) |
| `outputs/ablation_beta_lambda/` | Hyperparameter sensitivity (49 β × λ combinations) |
| `outputs/dense_retrieval/` | UniXcoder + FAISS retrieval metrics |
| `outputs/analysis/` | Error analysis and significance tests |
| `outputs/cceval_python_hf_bm25_full/` | CrossCodeEval BM25 results |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in values if using the API-based runner:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `STARCODER_API_URL` | Endpoint for remote model API (optional) |
| `STARCODER_API_KEY` | API key for remote model (optional) |
| `API_MODEL_NAME` | Model name for API backend (optional) |
