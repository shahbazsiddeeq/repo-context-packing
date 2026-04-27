"""Dense retrieval comparison using UniXcoder + FAISS (RQ5).

Evaluates 4 retrieval conditions on RepoEval (1600 examples):
  - BM25              (already done — loaded from existing CSV for reference)
  - BM25+Locality     (already done — loaded from existing CSV for reference)
  - Dense             (UniXcoder cosine similarity, top-10)
  - Dense+Locality    (Dense + locality reranking, β=150, λ=0.2)

Outputs:
  outputs/dense_retrieval/dense_retrieval_results.csv   — per-example metrics
  outputs/dense_retrieval/dense_retrieval_summary.json  — aggregate MRR/R@1/R@5
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel

from src.data.load_repoeval import load_repoeval_file
from src.chunking.fixed_chunker import chunk_record_fixed
from src.retrieval.locality_reranker import rerank_with_locality

from scripts.run_repoeval_first_experiment import (
    REPOEVAL_FILE,
    find_repo_dir,
    load_python_files_from_repo,
    find_gold_chunk_id,
)

# ── config ──────────────────────────────────────────────────────────────────
MAX_ROWS        = int(os.getenv("MAX_ROWS", "1600"))
TOP_K           = 10
MAX_FILES       = 1000
CHUNK_SIZE      = 120
CHUNK_OVERLAP   = 20
BETA            = 150.0
LAMBDA          = 0.2
BATCH_SIZE      = 64    # chunks per encoding batch
SAVE_EVERY      = 50    # checkpoint interval (examples)
MODEL_NAME      = "microsoft/unixcoder-base"
# Local snapshot that has model.safetensors (avoids torch.load CVE-2025-32434 block)
_HF_CACHE = Path.home() / ".cache/huggingface/hub"
_UNIXCODER_SNAPSHOT = (
    _HF_CACHE / "models--microsoft--unixcoder-base" / "snapshots"
    / "5604afdc964f6c53782a6813140ade5216b99006"
)
MODEL_LOCAL = str(_UNIXCODER_SNAPSHOT) if _UNIXCODER_SNAPSHOT.exists() else MODEL_NAME
OUTPUT_DIR      = Path(os.getenv("OUTPUT_DIR", "outputs/dense_retrieval"))
RESULTS_CSV     = OUTPUT_DIR / "dense_retrieval_results.csv"
SUMMARY_JSON    = OUTPUT_DIR / "dense_retrieval_summary.json"

CSV_FIELDS = [
    "task_id", "repo", "path",
    "rr_dense", "r1_dense", "r5_dense",
    "rr_dense_loc", "r1_dense_loc", "r5_dense_loc",
    "gold_found_dense", "gold_found_dense_loc",
    "num_chunks",
]


# ── tiny stand-in so locality reranker can consume dense results ─────────────
@dataclass
class DenseResult:
    chunk_id: str
    score: float
    rank: int
    text: str
    metadata: Dict[str, Any]


# ── metric helpers ───────────────────────────────────────────────────────────
def reciprocal_rank(results: List[Any], gold_id: str) -> float:
    for r in results:
        if r.chunk_id == gold_id:
            return 1.0 / r.rank
    return 0.0


def recall_at_k(results: List[Any], gold_id: str, k: int) -> float:
    return 1.0 if any(r.chunk_id == gold_id for r in results[:k]) else 0.0


# ── UniXcoder encoder ────────────────────────────────────────────────────────
class UniXcoderEncoder:
    def __init__(self, model_name: str, device: str):
        local = MODEL_LOCAL
        print(f"Loading tokenizer from: {local}")
        self.tokenizer = AutoTokenizer.from_pretrained(local)
        print(f"Loading model from: {local}")
        # use_safetensors=True avoids torch.load CVE-2025-32434 restriction
        self.model = AutoModel.from_pretrained(
            local, use_safetensors=True
        ).to(device)
        self.model.eval()
        self.device = device
        print(f"Model on {device}")

    def _encode_batch(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """Return L2-normalised embeddings for a batch of texts."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**encoded)

        # mean pool over non-padding tokens
        attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
        token_embs = out.last_hidden_state * attention_mask
        embs = token_embs.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
        embs = embs.cpu().float().numpy()

        # L2 normalise for cosine similarity via inner product
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.maximum(norms, 1e-9)
        return embs

    def encode(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        all_embs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            all_embs.append(self._encode_batch(batch))
        return np.vstack(all_embs)


# ── FAISS index ──────────────────────────────────────────────────────────────
def build_cosine_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Inner product on L2-normalised vectors = cosine similarity."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def dense_search(
    index: faiss.IndexFlatIP,
    query_emb: np.ndarray,
    chunks: List[Dict[str, Any]],
    top_k: int = TOP_K,
) -> List[DenseResult]:
    scores, indices = index.search(query_emb.reshape(1, -1).astype(np.float32), top_k)
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if idx < 0:
            continue
        chunk = chunks[idx]
        results.append(
            DenseResult(
                chunk_id=chunk["chunk_id"],
                score=float(score),
                rank=rank,
                text=chunk["text"],
                metadata={
                    "repo": chunk.get("repo"),
                    "path": chunk.get("path"),
                    "language": chunk.get("language"),
                    "start_line": chunk.get("start_line"),
                    "end_line": chunk.get("end_line"),
                },
            )
        )
    return results


# ── checkpoint helpers ───────────────────────────────────────────────────────
def load_done_ids(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    done = set()
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            done.add(row["task_id"])
    return done


def append_rows(csv_path: Path, rows: List[Dict], write_header: bool) -> None:
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    encoder = UniXcoderEncoder(MODEL_NAME, device)

    print(f"\nLoading {MAX_ROWS} RepoEval examples...")
    examples = load_repoeval_file(REPOEVAL_FILE, max_rows=MAX_ROWS)
    print(f"Loaded: {len(examples)} examples")

    done_ids = load_done_ids(RESULTS_CSV)
    if done_ids:
        print(f"Resuming — {len(done_ids)} examples already done")

    # aggregate accumulators — reload from checkpoint
    acc: Dict[str, float] = {k: 0.0 for k in ["rr_d", "r1_d", "r5_d", "rr_dl", "r1_dl", "r5_dl", "n"]}
    if done_ids and RESULTS_CSV.exists():
        with RESULTS_CSV.open() as f:
            for row in csv.DictReader(f):
                acc["rr_d"]  += float(row["rr_dense"])
                acc["r1_d"]  += float(row["r1_dense"])
                acc["r5_d"]  += float(row["r5_dense"])
                acc["rr_dl"] += float(row["rr_dense_loc"])
                acc["r1_dl"] += float(row["r1_dense_loc"])
                acc["r5_dl"] += float(row["r5_dense_loc"])
                acc["n"]     += 1

    # group examples by repo — encode chunks once per repo (8 repos × 200 examples)
    from collections import defaultdict
    by_repo: Dict[str, List[Dict]] = defaultdict(list)
    for ex in examples:
        by_repo[ex["repo"]].append(ex)

    repos = sorted(by_repo.keys())
    print(f"Repos: {len(repos)}")

    skipped = 0
    repo_num = 0

    for repo_name in repos:
        repo_examples = by_repo[repo_name]
        pending_in_repo = [e for e in repo_examples if e["task_id"] not in done_ids]

        repo_num += 1
        print(f"\n[repo {repo_num}/{len(repos)}] {repo_name} — "
              f"{len(pending_in_repo)}/{len(repo_examples)} examples to process")

        if not pending_in_repo:
            print("  all done, skipping")
            continue

        repo_dir = find_repo_dir(repo_name)
        if repo_dir is None:
            skipped += len(pending_in_repo)
            print(f"  [skip] repo dir not found")
            continue

        # build chunks once for the whole repo (no required_path — load everything)
        repo_records = load_python_files_from_repo(repo_dir, max_files=MAX_FILES)
        if not repo_records:
            skipped += len(pending_in_repo)
            print(f"  [skip] no python files")
            continue

        all_chunks = []
        for record in repo_records:
            if record["code"].strip():
                all_chunks.extend(
                    chunk_record_fixed(
                        record,
                        chunk_size_tokens=CHUNK_SIZE,
                        chunk_overlap_tokens=CHUNK_OVERLAP,
                    )
                )
        if not all_chunks:
            skipped += len(pending_in_repo)
            print(f"  [skip] no chunks after splitting")
            continue

        print(f"  chunks: {len(all_chunks)} — encoding...")
        chunk_texts = [c["text"] for c in all_chunks]
        chunk_embs  = encoder.encode(chunk_texts)
        index       = build_cosine_index(chunk_embs)
        print(f"  index built ({chunk_embs.shape[0]}×{chunk_embs.shape[1]})")

        pending_rows: List[Dict] = []

        for ex_idx, ex in enumerate(pending_in_repo, start=1):
            gold_id = find_gold_chunk_id(
                all_chunks, ex["path"], ex["line_no"], ex["ground_truth"]
            )
            if gold_id is None:
                skipped += 1
                continue

            query_emb     = encoder.encode([ex["query"]])
            dense_results = dense_search(index, query_emb[0], all_chunks, top_k=TOP_K)
            dense_loc_results = rerank_with_locality(
                dense_results,
                current_path=ex["path"],
                current_line=ex["line_no"] + 1,
                same_file_bonus=BETA,
                line_distance_weight=LAMBDA,
            )

            rr_d  = reciprocal_rank(dense_results, gold_id)
            r1_d  = recall_at_k(dense_results, gold_id, 1)
            r5_d  = recall_at_k(dense_results, gold_id, 5)
            rr_dl = reciprocal_rank(dense_loc_results, gold_id)
            r1_dl = recall_at_k(dense_loc_results, gold_id, 1)
            r5_dl = recall_at_k(dense_loc_results, gold_id, 5)

            acc["rr_d"]  += rr_d
            acc["r1_d"]  += r1_d
            acc["r5_d"]  += r5_d
            acc["rr_dl"] += rr_dl
            acc["r1_dl"] += r1_dl
            acc["r5_dl"] += r5_dl
            acc["n"]     += 1

            pending_rows.append({
                "task_id":              ex["task_id"],
                "repo":                 ex["repo"],
                "path":                 ex["path"],
                "rr_dense":             round(rr_d,  6),
                "r1_dense":             int(r1_d),
                "r5_dense":             int(r5_d),
                "rr_dense_loc":         round(rr_dl, 6),
                "r1_dense_loc":         int(r1_dl),
                "r5_dense_loc":         int(r5_dl),
                "gold_found_dense":     int(r1_d),
                "gold_found_dense_loc": int(r1_dl),
                "num_chunks":           len(all_chunks),
            })

            if ex_idx % 50 == 0:
                n = int(acc["n"])
                print(
                    f"  [{ex_idx}/{len(pending_in_repo)}] "
                    f"Dense MRR={acc['rr_d']/n:.4f} "
                    f"Dense+Loc MRR={acc['rr_dl']/n:.4f}"
                )

        # checkpoint after each repo
        if pending_rows:
            write_header = not RESULTS_CSV.exists()
            append_rows(RESULTS_CSV, pending_rows, write_header=write_header)
            n = int(acc["n"])
            print(f"  [checkpoint] {len(pending_rows)} rows saved | total={n}")

    n = int(acc["n"])
    print(f"\n{'='*70}")
    print(f"Usable: {n}/{len(examples)}  Skipped: {skipped}")
    print(f"{'='*70}")
    if n > 0:
        summary = {
            "model":    MODEL_NAME,
            "max_rows": MAX_ROWS,
            "usable":   n,
            "skipped":  skipped,
            "beta":     BETA,
            "lambda":   LAMBDA,
            "Dense": {
                "MRR":      round(acc["rr_d"]  / n, 6),
                "Recall@1": round(acc["r1_d"]  / n, 6),
                "Recall@5": round(acc["r5_d"]  / n, 6),
            },
            "Dense+Locality": {
                "MRR":      round(acc["rr_dl"] / n, 6),
                "Recall@1": round(acc["r1_dl"] / n, 6),
                "Recall@5": round(acc["r5_dl"] / n, 6),
            },
        }
        print(f"\n{'Condition':<18} {'MRR':>8} {'R@1':>8} {'R@5':>8}")
        print("-" * 46)
        for cond, m in summary.items():
            if isinstance(m, dict):
                print(f"{cond:<18} {m['MRR']:>8.4f} {m['Recall@1']:>8.4f} {m['Recall@5']:>8.4f}")

        with SUMMARY_JSON.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {RESULTS_CSV}")
        print(f"Saved: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
