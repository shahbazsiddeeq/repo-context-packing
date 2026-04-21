from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.data.load_repoeval import load_repoeval_file
from src.chunking.fixed_chunker import chunk_record_fixed
from src.chunking.structure_chunker import chunk_record_structure_aware
from src.chunking.hybrid_chunker import chunk_record_hybrid
from src.retrieval.bm25_retriever import BM25CodeRetriever
from src.retrieval.locality_reranker import rerank_with_locality

from scripts.run_repoeval_first_experiment import (
    REPOEVAL_FILE,
    find_repo_dir,
    load_python_files_from_repo,
    find_gold_chunk_id,
)

MAX_ROWS = int(os.getenv("MAX_ROWS", "200"))
TOP_K = 10
MAX_FILES_PER_REPO = 1000
SAME_FILE_BONUS = 150.0
LINE_DISTANCE_WEIGHT = 0.2

CHUNK_SIZE = 120
CHUNK_OVERLAP = 20
STRUCTURE_MAX_TOKENS = 140
STRUCTURE_OVERLAP = 20

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", f"outputs/chunking_comparison_{MAX_ROWS}"))


def reciprocal_rank(results, gold_chunk_id: str) -> float:
    for r in results:
        if r.chunk_id == gold_chunk_id:
            return 1.0 / r.rank
    return 0.0


def recall_at_k(results, gold_chunk_id: str, k: int) -> float:
    return 1.0 if any(r.chunk_id == gold_chunk_id for r in results[:k]) else 0.0


def build_chunks(repo_records: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []
    for record in repo_records:
        if not record["code"].strip():
            continue
        if mode == "fixed":
            chunks = chunk_record_fixed(
                record,
                chunk_size_tokens=CHUNK_SIZE,
                chunk_overlap_tokens=CHUNK_OVERLAP,
            )
        elif mode == "structure":
            chunks = chunk_record_structure_aware(
                record,
                fallback_to_fixed=True,
                fixed_chunk_size_tokens=CHUNK_SIZE,
                fixed_chunk_overlap_tokens=CHUNK_OVERLAP,
                max_chunk_tokens=STRUCTURE_MAX_TOKENS,
                overlap_tokens=STRUCTURE_OVERLAP,
            )
        elif mode == "hybrid":
            chunks = chunk_record_hybrid(
                record,
                fixed_chunk_size_tokens=CHUNK_SIZE,
                fixed_chunk_overlap_tokens=CHUNK_OVERLAP,
                structure_max_chunk_tokens=STRUCTURE_MAX_TOKENS,
                structure_overlap_tokens=STRUCTURE_OVERLAP,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        all_chunks.extend(chunks)
    return all_chunks


def run_mode(
    examples: List[Dict[str, Any]],
    mode: str,
    use_locality: bool = False,
) -> Dict[str, Any]:
    label = f"{mode}+locality" if use_locality else mode

    rr_list, r1_list, r5_list = [], [], []
    per_example_rows = []
    usable = skipped_repo = skipped_files = skipped_chunks = skipped_gold = 0

    for idx, ex in enumerate(examples, start=1):
        repo_name = ex["repo"]
        target_path = ex["path"]
        target_line = ex["line_no"]
        query = ex["query"]

        if idx % 50 == 0 or idx == 1:
            print(f"  [{label}] {idx}/{len(examples)}")

        repo_dir = find_repo_dir(repo_name)
        if repo_dir is None:
            skipped_repo += 1
            continue

        repo_records = load_python_files_from_repo(
            repo_dir, max_files=MAX_FILES_PER_REPO, required_path=target_path
        )
        if not repo_records:
            skipped_files += 1
            continue

        all_chunks = build_chunks(repo_records, mode)
        if not all_chunks:
            skipped_chunks += 1
            continue

        gold_chunk_id = find_gold_chunk_id(
            all_chunks, target_path, target_line, ex["ground_truth"]
        )
        if gold_chunk_id is None:
            skipped_gold += 1
            continue

        retriever = BM25CodeRetriever(all_chunks)
        results = retriever.search(query, top_k=TOP_K)

        if use_locality:
            results = rerank_with_locality(
                results,
                current_path=target_path,
                current_line=target_line + 1,
                same_file_bonus=SAME_FILE_BONUS,
                line_distance_weight=LINE_DISTANCE_WEIGHT,
            )

        rr = reciprocal_rank(results, gold_chunk_id)
        r1 = recall_at_k(results, gold_chunk_id, 1)
        r5 = recall_at_k(results, gold_chunk_id, 5)

        rr_list.append(rr)
        r1_list.append(r1)
        r5_list.append(r5)
        usable += 1

        per_example_rows.append({
            "task_id": ex["task_id"],
            "repo": repo_name,
            "path": target_path,
            "line_no": target_line,
            "mode": label,
            "rr": rr,
            "r1": r1,
            "r5": r5,
        })

    if usable == 0:
        raise RuntimeError(f"No usable examples for mode={label}")

    mrr = sum(rr_list) / usable
    recall1 = sum(r1_list) / usable
    recall5 = sum(r5_list) / usable

    print(f"\n  [{label}] usable={usable} MRR={mrr:.4f} R@1={recall1:.4f} R@5={recall5:.4f}")
    print(f"  skipped: repo={skipped_repo} files={skipped_files} chunks={skipped_chunks} gold={skipped_gold}")

    return {
        "mode": label,
        "examples_requested": len(examples),
        "usable": usable,
        "MRR": mrr,
        "Recall@1": recall1,
        "Recall@5": recall5,
        "skipped_repo": skipped_repo,
        "skipped_files": skipped_files,
        "skipped_chunks": skipped_chunks,
        "skipped_gold": skipped_gold,
        "per_example": per_example_rows,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MAX_ROWS} RepoEval examples...")
    examples = load_repoeval_file(REPOEVAL_FILE, max_rows=MAX_ROWS)
    print(f"Loaded: {len(examples)} examples\n")

    modes_to_run = [
        ("fixed", False),
        ("fixed", True),
        ("structure", False),
        ("hybrid", False),
    ]

    all_summaries = []
    all_per_example = []

    for mode, use_locality in modes_to_run:
        print(f"=== Running: {mode} {'+ locality' if use_locality else ''} ===")
        result = run_mode(examples, mode, use_locality)
        summary = {k: v for k, v in result.items() if k != "per_example"}
        all_summaries.append(summary)
        all_per_example.extend(result["per_example"])

    # Save summary CSV
    summary_path = OUTPUT_DIR / "chunking_comparison_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(all_summaries)

    # Save per-example CSV
    per_example_path = OUTPUT_DIR / "chunking_comparison_per_example.csv"
    with per_example_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_per_example[0].keys()))
        writer.writeheader()
        writer.writerows(all_per_example)

    # Save JSON
    json_path = OUTPUT_DIR / "chunking_comparison_summary.json"
    with json_path.open("w") as f:
        json.dump(all_summaries, f, indent=2)

    # Print table
    print("\n" + "=" * 70)
    print(f"Chunking Comparison — {MAX_ROWS} examples")
    print(f"{'Method':<25} {'MRR':>8} {'R@1':>8} {'R@5':>8} {'Usable':>8}")
    print("-" * 70)
    for s in all_summaries:
        print(f"{s['mode']:<25} {s['MRR']:>8.4f} {s['Recall@1']:>8.4f} {s['Recall@5']:>8.4f} {s['usable']:>8}")
    print("=" * 70)

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {per_example_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
