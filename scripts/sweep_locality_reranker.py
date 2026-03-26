from __future__ import annotations

from typing import Dict, Any, List, Tuple

from src.data.load_repoeval import load_repoeval_file
from src.chunking.fixed_chunker import chunk_record_fixed
from src.retrieval.bm25_retriever import BM25CodeRetriever
from src.retrieval.locality_reranker import rerank_with_locality

from scripts.run_repoeval_first_experiment import (
    REPOEVAL_FILE,
    find_repo_dir,
    load_python_files_from_repo,
    find_gold_chunk_id,
)

PARAMS: List[Tuple[float, float]] = [
    (50.0, 0.1),
    (100.0, 0.2),
    (150.0, 0.2),
    (100.0, 0.5),
]


def reciprocal_rank(results, gold_chunk_id: str) -> float:
    for r in results:
        if r.chunk_id == gold_chunk_id:
            return 1.0 / r.rank
    return 0.0


def recall_at_k(results, gold_chunk_id: str, k: int) -> float:
    return 1.0 if any(r.chunk_id == gold_chunk_id for r in results[:k]) else 0.0


def build_fixed_chunks(repo_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []

    for record in repo_records:
        if not record["code"].strip():
            continue

        chunks = chunk_record_fixed(
            record,
            code_field="code",
            repo_field="repo",
            path_field="path",
            language_field="language",
            chunk_size_tokens=120,
            chunk_overlap_tokens=20,
        )
        all_chunks.extend(chunks)

    return all_chunks


def init_metrics() -> Dict[str, float]:
    return {
        "usable_examples": 0,
        "rr_sum": 0.0,
        "r1_sum": 0.0,
        "r5_sum": 0.0,
    }


def finalize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    n = metrics["usable_examples"]
    if n == 0:
        return {
            "usable_examples": 0,
            "MRR": 0.0,
            "Recall@1": 0.0,
            "Recall@5": 0.0,
        }

    return {
        "usable_examples": int(n),
        "MRR": metrics["rr_sum"] / n,
        "Recall@1": metrics["r1_sum"] / n,
        "Recall@5": metrics["r5_sum"] / n,
    }


def print_summary(title: str, metrics: Dict[str, float]) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("-" * 100)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:16s}: {v:.4f}")
        else:
            print(f"{k:16s}: {v}")


def main():
    examples = load_repoeval_file(REPOEVAL_FILE, max_rows=100)

    # baseline once
    baseline_metrics = init_metrics()

    rerank_metrics_by_param = {
        (bonus, weight): init_metrics() for (bonus, weight) in PARAMS
    }

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

        all_chunks = build_fixed_chunks(repo_records)
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
        baseline_results = retriever.search(query, top_k=10)

        baseline_metrics["usable_examples"] += 1
        baseline_metrics["rr_sum"] += reciprocal_rank(baseline_results, gold_chunk_id)
        baseline_metrics["r1_sum"] += recall_at_k(baseline_results, gold_chunk_id, 1)
        baseline_metrics["r5_sum"] += recall_at_k(baseline_results, gold_chunk_id, 5)

        for bonus, weight in PARAMS:
            reranked_results = rerank_with_locality(
                baseline_results,
                current_path=target_path,
                current_line=target_line + 1,
                same_file_bonus=bonus,
                line_distance_weight=weight,
            )

            m = rerank_metrics_by_param[(bonus, weight)]
            m["usable_examples"] += 1
            m["rr_sum"] += reciprocal_rank(reranked_results, gold_chunk_id)
            m["r1_sum"] += recall_at_k(reranked_results, gold_chunk_id, 1)
            m["r5_sum"] += recall_at_k(reranked_results, gold_chunk_id, 5)

    print_summary("Fixed BM25 baseline", finalize_metrics(baseline_metrics))

    for bonus, weight in PARAMS:
        print_summary(
            f"Locality reranking (same_file_bonus={bonus}, line_distance_weight={weight})",
            finalize_metrics(rerank_metrics_by_param[(bonus, weight)]),
        )


if __name__ == "__main__":
    main()