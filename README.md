# Locality-Aware Reranking for Repository-Level Code Completion

Replication package for the Computing 2027 paper on locality-aware reranking for repository-level code completion.

## Method

For each retrieved chunk `c`, the reranked score is

```
s'(c) = s_ret(c) + I[c.path = f] · (β − λ · d(c, l))
```

- `β = 150.0`: same-file bonus; `λ = 0.2`: line-distance weight
- `d(c, l)`: line distance from the chunk span to the target line `l`

Implementation: `src/retrieval/locality_reranker.py`.

**Indexing protocol.** The target
file is cut to lines `1 … l−1` before chunking (`truncate_line` in
`scripts/run_repoeval_first_experiment.py::load_python_files_from_repo`), so no chunk
contains the reference line or any later line. All other files are indexed in full.

---

## Results

### Evaluation (Section 5)

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

### Five-condition run (GPU with ~24 GB)

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
python scripts/fig_clean_results.py outputs/CR_clean_merged/CR_clean_merged_per_example.csv figures/   # Figs. 2-4

# Prompt-chunk analysis (Section 5.3, Tables 3-4): ES, overlap with the prefix,
# missing identifiers, kind of reference line, copying
python scripts/extract_prompt_chunks.py outputs/CR_clean_merged/CR_clean_merged_per_example.csv \
    outputs/CR_clean_analysis/prompt_chunks.jsonl 8
python scripts/analyze_clean_chunks.py outputs/CR_clean_merged/CR_clean_merged_per_example.csv \
    outputs/CR_clean_analysis/prompt_chunks.jsonl outputs/CR_clean_analysis/chunk_analysis.json
```

### CrossCodeEval (Section 5.1; uses the benchmark's own cross-file retrieval files)

```bash
python scripts/run_crosscodeeval_official_prompts.py
```


---

## Data

[RepoEval](https://github.com/microsoft/CodeT/tree/main/RepoCoder) line-level split
(1,600 Python instances from 8 repositories). Download instructions: `scripts/data_prep/`.

---

## Pre-computed outputs

### RepoEval runs (Section 5)

| Directory | Contents |
|---|---|
| `outputs/CR_clean_merged/` | All 1,600 instances, five conditions, and `clean_run_stats.json` |
| `outputs/CR_clean_repocoder_starcoder2_7b_1600/` | Instances 0–799 |
| `outputs/CR_clean_repocoder_starcoder2_7b_1600_part2/` | Instances 800–1599 |
| `outputs/CR_clean_analysis/chunk_analysis.json` | Prompt-chunk analysis behind Tables 3 and 4 (`prompt_chunks.jsonl` is regenerated, not stored) |
| `outputs/clean_completion_starcoder2_7b_1600/` | Pilot with an uncommented prompt format (Section 7) |

### CrossCodeEval (Section 5.1)

| Directory | Contents |
|---|---|
| `outputs/cceval_python_hf_bm25_full/`, `outputs/cceval_python_hf_promptonly_full/` | StarCoder2-7B with and without the provided cross-file context |

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

## License

Code in this repository is released under the [MIT License](LICENSE). The RepoEval and
CrossCodeEval datasets are not included. The per-instance output files contain excerpts
from them (reference lines and retrieved code), which remain under the benchmarks' own licenses.
