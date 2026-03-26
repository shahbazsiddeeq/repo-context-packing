from __future__ import annotations

from typing import Dict, Any, List

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
    examples = load_repoeval_file(REPOEVAL_FILE, max_rows=20)

    baseline_metrics = init_metrics()
    reranked_metrics = init_metrics()

    for ex in examples:
        repo_name = ex["repo"]
        target_path = ex["path"]
        target_line = ex["line_no"]
        query = ex["query"]
        ground_truth = ex["ground_truth"]

        repo_dir = find_repo_dir(repo_name)
        if repo_dir is None:
            print(f"[skip] repo not found: {repo_name}")
            continue

        repo_records = load_python_files_from_repo(
            repo_dir,
            max_files=200,
            required_path=target_path,
        )
        if not repo_records:
            print(f"[skip] no python files: {repo_name}")
            continue

        all_chunks = build_fixed_chunks(repo_records)
        if not all_chunks:
            print(f"[skip] no chunks: {repo_name}")
            continue

        gold_chunk_id = find_gold_chunk_id(
            all_chunks,
            target_path,
            target_line,
            ground_truth,
        )
        if gold_chunk_id is None:
            print(f"[skip] gold not matched: {repo_name} | {target_path}:{target_line}")
            continue

        retriever = BM25CodeRetriever(all_chunks)
        baseline_results = retriever.search(query, top_k=10)

        # RepoEval line-level behaves like insertion-at-line_no,
        # while the missing target line is often at line_no + 1.
        reranked_results = rerank_with_locality(
            baseline_results,
            current_path=target_path,
            current_line=target_line + 1,
            same_file_bonus=100.0,
            line_distance_weight=0.2,
        )

        # baseline metrics
        baseline_metrics["usable_examples"] += 1
        baseline_metrics["rr_sum"] += reciprocal_rank(baseline_results, gold_chunk_id)
        baseline_metrics["r1_sum"] += recall_at_k(baseline_results, gold_chunk_id, 1)
        baseline_metrics["r5_sum"] += recall_at_k(baseline_results, gold_chunk_id, 5)

        # reranked metrics
        reranked_metrics["usable_examples"] += 1
        reranked_metrics["rr_sum"] += reciprocal_rank(reranked_results, gold_chunk_id)
        reranked_metrics["r1_sum"] += recall_at_k(reranked_results, gold_chunk_id, 1)
        reranked_metrics["r5_sum"] += recall_at_k(reranked_results, gold_chunk_id, 5)

        print("\n" + "#" * 100)
        print(f"task_id      : {ex['task_id']}")
        print(f"path         : {target_path}")
        print(f"line_no      : {target_line}")
        print(f"gold_chunk   : {gold_chunk_id}")

        if baseline_results:
            print(
                f"baseline top1: {baseline_results[0].chunk_id} "
                f"(path={baseline_results[0].metadata.get('path')} "
                f"lines={baseline_results[0].metadata.get('start_line')}-{baseline_results[0].metadata.get('end_line')})"
            )

        if reranked_results:
            print(
                f"rerank top1  : {reranked_results[0].chunk_id} "
                f"(path={reranked_results[0].metadata.get('path')} "
                f"lines={reranked_results[0].metadata.get('start_line')}-{reranked_results[0].metadata.get('end_line')})"
            )

        print(
            f"baseline  -> RR={reciprocal_rank(baseline_results, gold_chunk_id):.4f}, "
            f"R@1={recall_at_k(baseline_results, gold_chunk_id, 1):.1f}, "
            f"R@5={recall_at_k(baseline_results, gold_chunk_id, 5):.1f}"
        )
        print(
            f"locality  -> RR={reciprocal_rank(reranked_results, gold_chunk_id):.4f}, "
            f"R@1={recall_at_k(reranked_results, gold_chunk_id, 1):.1f}, "
            f"R@5={recall_at_k(reranked_results, gold_chunk_id, 5):.1f}"
        )

    print_summary("Fixed BM25 baseline", finalize_metrics(baseline_metrics))
    print_summary("Fixed BM25 + locality reranking", finalize_metrics(reranked_metrics))


if __name__ == "__main__":
    main()