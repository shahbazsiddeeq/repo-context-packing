from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

from src.data.load_repoeval import load_repoeval_file
from src.chunking.structure_chunker import chunk_record_structure_aware
from src.retrieval.bm25_retriever import BM25CodeRetriever

from scripts.run_repoeval_first_experiment import (
    REPOEVAL_FILE,
    find_repo_dir,
    load_python_files_from_repo,
    reciprocal_rank,
    recall_at_k,
    find_gold_chunk_id,
)


CHUNK_SIZES = [60, 100, 140, 200]


def build_structure_chunks(
    repo_records: List[Dict[str, Any]],
    max_chunk_tokens: int,
    overlap_tokens: int = 20,
) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []

    for record in repo_records:
        if not record["code"].strip():
            continue

        chunks = chunk_record_structure_aware(
            record,
            code_field="code",
            repo_field="repo",
            path_field="path",
            language_field="language",
            fallback_to_fixed=True,
            fixed_chunk_size_tokens=120,
            fixed_chunk_overlap_tokens=20,
            max_chunk_tokens=max_chunk_tokens,
            overlap_tokens=overlap_tokens,
        )
        all_chunks.extend(chunks)

    return all_chunks


def evaluate_structure_size(
    max_chunk_tokens: int,
    max_examples: int = 5,
    top_k: int = 10,
) -> Dict[str, Any]:
    examples = load_repoeval_file(REPOEVAL_FILE, max_rows=max_examples)

    rr_scores = []
    r1_scores = []
    r5_scores = []
    usable = 0

    for ex in examples:
        repo_name = ex["repo"]
        target_path = ex["path"]
        target_line = ex["line_no"]
        query = ex["query"]
        ground_truth = ex["ground_truth"]

        repo_dir = find_repo_dir(repo_name)
        if repo_dir is None:
            continue

        repo_records = load_python_files_from_repo(
            repo_dir,
            max_files=200,
            required_path=target_path,
        )
        if not repo_records:
            continue

        all_chunks = build_structure_chunks(
            repo_records,
            max_chunk_tokens=max_chunk_tokens,
            overlap_tokens=20,
        )
        if not all_chunks:
            continue

        gold_chunk_id = find_gold_chunk_id(
            all_chunks,
            target_path,
            target_line,
            ground_truth,
        )
        if gold_chunk_id is None:
            continue

        retriever = BM25CodeRetriever(all_chunks)
        results = retriever.search(query, top_k=top_k)

        rr_scores.append(reciprocal_rank(results, gold_chunk_id))
        r1_scores.append(recall_at_k(results, gold_chunk_id, 1))
        r5_scores.append(recall_at_k(results, gold_chunk_id, 5))
        usable += 1

    if usable == 0:
        return {
            "chunk_size": max_chunk_tokens,
            "usable_examples": 0,
            "MRR": 0.0,
            "Recall@1": 0.0,
            "Recall@5": 0.0,
        }

    return {
        "chunk_size": max_chunk_tokens,
        "usable_examples": usable,
        "MRR": sum(rr_scores) / usable,
        "Recall@1": sum(r1_scores) / usable,
        "Recall@5": sum(r5_scores) / usable,
    }


def print_summary(metrics: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(f"Structure-aware chunk size = {metrics['chunk_size']}")
    print("-" * 80)
    print(f"usable_examples : {metrics['usable_examples']}")
    print(f"MRR             : {metrics['MRR']:.4f}")
    print(f"Recall@1        : {metrics['Recall@1']:.4f}")
    print(f"Recall@5        : {metrics['Recall@5']:.4f}")


def main():
    for size in CHUNK_SIZES:
        metrics = evaluate_structure_size(
            max_chunk_tokens=size,
            max_examples=5,
            top_k=10,
        )
        print_summary(metrics)


if __name__ == "__main__":
    main()