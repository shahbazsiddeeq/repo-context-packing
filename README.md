# Locality-Aware Reranking for Repository-Level Code Completion: Target-File Leakage and a Clean Re-Evaluation

Replication package for the paper of the same title (Computing 2027).

> **Important.** Earlier versions of this repository reported exact-match and MRR gains
> for locality-aware reranking. Those runs built the retrieval index from the **full**
> target file, so a same-file chunk could contain the reference line. Those gains are
> leakage artifacts (fix: commit "Fix RepoEval answer leakage: truncate target file at
> completion point"). The outputs of those runs are kept for the leakage analysis and
> are labelled **untruncated** below. They must not be read as evidence that the method works.

## Method

For each retrieved chunk `c`, the reranked score is

```
s'(c) = s_ret(c) + I[c.path = f] · (β − λ · d(c, l))
```

- `β = 150.0`: same-file bonus; `λ = 0.2`: line-distance weight
- `d(c, l)`: line distance from the chunk span to the target line `l`

Implementation: `src/retrieval/locality_reranker.py`.

**Indexing protocol.** In the truncated protocol (used for all valid results), the target
file is cut to lines `1 … l−1` before chunking (`truncate_line` in
`scripts/run_repoeval_first_experiment.py::load_python_files_from_repo`), so no chunk
contains the reference line or any later line. All other files are indexed in full.

---

## Results

### Clean re-evaluation (truncated index)

StarCoder2-7B, all 1,600 RepoEval line-level instances, RepoCoder prompt format
(retrieved chunk as commented lines before the prefix), top-1 context.

| Condition | EM (%) |
|---|---|
| Prompt-only (no context) | 34.62 |
| BM25 top-1 | 30.25 |
| Locality top-1 | 29.31 |
| Nearest preceding same-file window (no retriever) | 28.88 |
| Highest-ranked cross-file BM25 chunk | **45.06** |

- Locality vs. BM25: −0.94 pp, 2 gains / 17 losses, exact McNemar p = 7.3×10⁻⁴.
- Locality vs. nearest window: +0.44 pp, p = 0.26. The locality top-1 chunk *is* the nearest window in 1,284 / 1,600 instances.
- Cross-file vs. prompt-only: +10.44 pp, 190 gains / 23 losses.

Full statistics: `outputs/CR_clean_merged/clean_run_stats.json`.

### Leakage analysis of the untruncated runs

| | Count |
|---|---|
| BM25 top-1 chunk contains the reference line | 1,081 / 1,600 |
| Locality top-1 chunk contains the reference line | 1,179 / 1,600 |
| Locality gains (5 models pooled) where reranking newly exposed the reference line | 163 / 169 |
| Locality gains / losses where neither top-1 chunk contains it (5 models pooled) | 6 / 51 |

Per-model values: `outputs/leakage_analysis/leakage_analysis.json`.

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `pyyaml`, `tqdm`, `rank-bm25`, `scikit-learn`, `matplotlib`,
`tree-sitter`, `scipy`, `torch`, `transformers`, `accelerate`, `faiss-cpu`.
The runs used Python 3.11, torch 2.4.1 (CUDA 12.1) and transformers 5.5.4.

Set `PYTHONPATH` to the repository root before running any script.

---

## Reproducing the paper

### Clean run (truncated index, five conditions; GPU with ~24 GB)

```bash
bash scripts/run_camera_ready_clean.sh            # all 1,600 instances
# or split across two GPUs:
#   ... --start_row 0   --end_row 800  --output_dir outputs/CR_clean_repocoder_starcoder2_7b_1600
#   ... --start_row 800 --end_row 1600 --output_dir outputs/CR_clean_repocoder_starcoder2_7b_1600_part2
```

The runner checkpoints every 50 instances and resumes if restarted.

### Statistics

```bash
python scripts/analyze_clean_run.py outputs/CR_clean_merged/CR_clean_merged_per_example.csv
python scripts/analyze_leakage.py
```

### CrossCodeEval (appendix; uses the benchmark's own cross-file retrieval files)

```bash
python scripts/run_crosscodeeval_official_prompts.py
```

The remaining scripts (`run_repoeval_full_locality.py`, `run_dense_retrieval.py`,
`run_dense_completion_model.py`, `sweep_locality_reranker.py`,
`run_chunking_comparison_full.py`) produced the **untruncated** runs and do not truncate
the target file.

---

## Data

[RepoEval](https://github.com/microsoft/CodeT/tree/main/RepoCoder) line-level split
(1,600 Python instances from 8 repositories). Download instructions: `scripts/data_prep/`.

---

## Pre-computed outputs

### Valid (truncated index)

| Directory | Contents |
|---|---|
| `outputs/CR_clean_merged/` | Clean run, all 1,600 instances, five conditions + `clean_run_stats.json` |
| `outputs/CR_clean_repocoder_starcoder2_7b_1600/` | Clean run, instances 0–799 |
| `outputs/CR_clean_repocoder_starcoder2_7b_1600_part2/` | Clean run, instances 800–1599 |
| `outputs/clean_completion_starcoder2_7b_1600/` | Earlier truncated run with an uncommented prompt format; the model copies the `# path` header line in 506+ outputs, so it is reported only as a prompt-format check |
| `outputs/leakage_analysis/` | Exposure analysis of the untruncated runs |
| `outputs/cceval_python_hf_bm25_full/`, `outputs/cceval_python_hf_promptonly_full/` | CrossCodeEval (not affected by the target-file problem) |

### Untruncated index (leaky; kept for the leakage analysis only)

| Directory | Contents |
|---|---|
| `outputs/completion_step1_1600/` | StarCoder2-7B |
| `outputs/completion_starcoder2_3b_1600/` | StarCoder2-3B |
| `outputs/completion_codellama_7b_1600/` | CodeLlama-7B |
| `outputs/completion_deepseek_6b_1600/` | DeepSeek-Coder-6.7B |
| `outputs/completion_codeqwen_7b_1600/` | CodeQwen1.5-7B |
| `outputs/completion_dense_starcoder2_7b/` | Dense (UniXcoder) + locality |
| `outputs/chunking_comparison_1600/`, `outputs/ablation_beta_lambda/`, `outputs/dense_retrieval/`, `outputs/analysis/` | Retrieval metrics, β×λ grid, error analysis |

---

## Project structure

```
src/
├── chunking/       fixed-size, structure-aware (tree-sitter), hybrid
├── retrieval/      BM25 retriever + locality reranker
├── packing/        top-k, neighbour, dependency-aware context packing
├── evaluation/     retrieval and completion metrics
└── data/           RepoEval and CodeSearchNet loaders
scripts/            experiment runners and analysis scripts
outputs/            pre-computed per-instance results
```

## Environment variables

Only needed for the API-based runner: copy `.env.example` to `.env` and fill in
`STARCODER_API_URL`, `STARCODER_API_KEY`, `API_MODEL_NAME`.
