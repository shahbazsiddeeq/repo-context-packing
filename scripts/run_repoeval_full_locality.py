from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Any, List, Optional

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

# Best setting from your sweeps
SAME_FILE_BONUS = 150.0
LINE_DISTANCE_WEIGHT = 0.2
TOP_K = 10

# Full run: None means do not truncate benchmark rows
MAX_ROWS: Optional[int] = None

# Increase repository coverage compared with earlier pilot runs
MAX_FILES_PER_REPO = 1000

OUTPUT_DIR = Path("outputs/full_run")
PER_EXAMPLE_CSV = OUTPUT_DIR / "repoeval_full_locality_per_example.csv"
SUMMARY_CSV = OUTPUT_DIR / "repoeval_full_locality_summary.csv"
REPORT_TXT = OUTPUT_DIR / "repoeval_full_locality_report.txt"


def reciprocal_rank(results, gold_chunk_id: str) -> float:
    for r in results:
        if r.chunk_id == gold_chunk_id:
            return 1.0 / r.rank
    return 0.0


def recall_at_k(results, gold_chunk_id: str, k: int) -> float:
    return 1.0 if any(r.chunk_id == gold_chunk_id for r in results[:k]) else 0.0


def first_rank(results, gold_chunk_id: str) -> int:
    for r in results:
        if r.chunk_id == gold_chunk_id:
            return r.rank
    return -1


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


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to save for {path}")

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    examples = load_repoeval_file(REPOEVAL_FILE, max_rows=MAX_ROWS)

    per_example_rows: List[Dict[str, Any]] = []
    baseline_metrics = init_metrics()
    reranked_metrics = init_metrics()

    total_examples = len(examples)
    skipped_repo = 0
    skipped_files = 0
    skipped_chunks = 0
    skipped_gold = 0

    print(f"Loaded benchmark examples: {total_examples}")

    for idx, ex in enumerate(examples, start=1):
        repo_name = ex["repo"]
        target_path = ex["path"]
        target_line = ex["line_no"]
        query = ex["query"]
        ground_truth = ex["ground_truth"]

        if idx % 25 == 0:
            print(f"Processing example {idx}/{total_examples} ...")

        repo_dir = find_repo_dir(repo_name)
        if repo_dir is None:
            skipped_repo += 1
            continue

        repo_records = load_python_files_from_repo(
            repo_dir,
            max_files=MAX_FILES_PER_REPO,
            required_path=target_path,
        )
        if not repo_records:
            skipped_files += 1
            continue

        all_chunks = build_fixed_chunks(repo_records)
        if not all_chunks:
            skipped_chunks += 1
            continue

        gold_chunk_id = find_gold_chunk_id(
            all_chunks,
            target_path,
            target_line,
            ground_truth,
        )
        if gold_chunk_id is None:
            skipped_gold += 1
            continue

        retriever = BM25CodeRetriever(all_chunks)
        baseline_results = retriever.search(query, top_k=TOP_K)

        reranked_results = rerank_with_locality(
            baseline_results,
            current_path=target_path,
            current_line=target_line + 1,
            same_file_bonus=SAME_FILE_BONUS,
            line_distance_weight=LINE_DISTANCE_WEIGHT,
        )

        baseline_metrics["usable_examples"] += 1
        baseline_metrics["rr_sum"] += reciprocal_rank(baseline_results, gold_chunk_id)
        baseline_metrics["r1_sum"] += recall_at_k(baseline_results, gold_chunk_id, 1)
        baseline_metrics["r5_sum"] += recall_at_k(baseline_results, gold_chunk_id, 5)

        reranked_metrics["usable_examples"] += 1
        reranked_metrics["rr_sum"] += reciprocal_rank(reranked_results, gold_chunk_id)
        reranked_metrics["r1_sum"] += recall_at_k(reranked_results, gold_chunk_id, 1)
        reranked_metrics["r5_sum"] += recall_at_k(reranked_results, gold_chunk_id, 5)

        baseline_top1 = baseline_results[0] if baseline_results else None
        reranked_top1 = reranked_results[0] if reranked_results else None

        per_example_rows.append(
            {
                "task_id": ex["task_id"],
                "repo": repo_name,
                "path": target_path,
                "line_no": target_line,
                "gold_chunk_id": gold_chunk_id,

                "baseline_gold_rank": first_rank(baseline_results, gold_chunk_id),
                "baseline_rr": reciprocal_rank(baseline_results, gold_chunk_id),
                "baseline_r1": recall_at_k(baseline_results, gold_chunk_id, 1),
                "baseline_r5": recall_at_k(baseline_results, gold_chunk_id, 5),
                "baseline_top1_chunk": baseline_top1.chunk_id if baseline_top1 else "",
                "baseline_top1_path": baseline_top1.metadata.get("path", "") if baseline_top1 else "",
                "baseline_top1_start": baseline_top1.metadata.get("start_line", -1) if baseline_top1 else -1,
                "baseline_top1_end": baseline_top1.metadata.get("end_line", -1) if baseline_top1 else -1,

                "rerank_gold_rank": first_rank(reranked_results, gold_chunk_id),
                "rerank_rr": reciprocal_rank(reranked_results, gold_chunk_id),
                "rerank_r1": recall_at_k(reranked_results, gold_chunk_id, 1),
                "rerank_r5": recall_at_k(reranked_results, gold_chunk_id, 5),
                "rerank_top1_chunk": reranked_top1.chunk_id if reranked_top1 else "",
                "rerank_top1_path": reranked_top1.metadata.get("path", "") if reranked_top1 else "",
                "rerank_top1_start": reranked_top1.metadata.get("start_line", -1) if reranked_top1 else -1,
                "rerank_top1_end": reranked_top1.metadata.get("end_line", -1) if reranked_top1 else -1,
            }
        )

    baseline_summary = finalize_metrics(baseline_metrics)
    reranked_summary = finalize_metrics(reranked_metrics)

    summary_rows = [
        {
            "method": "fixed_bm25",
            "usable_examples": baseline_summary["usable_examples"],
            "MRR": baseline_summary["MRR"],
            "Recall@1": baseline_summary["Recall@1"],
            "Recall@5": baseline_summary["Recall@5"],
            "same_file_bonus": "",
            "line_distance_weight": "",
            "top_k": TOP_K,
            "max_rows": "full" if MAX_ROWS is None else MAX_ROWS,
            "max_files_per_repo": MAX_FILES_PER_REPO,
        },
        {
            "method": "fixed_bm25_locality_rerank",
            "usable_examples": reranked_summary["usable_examples"],
            "MRR": reranked_summary["MRR"],
            "Recall@1": reranked_summary["Recall@1"],
            "Recall@5": reranked_summary["Recall@5"],
            "same_file_bonus": SAME_FILE_BONUS,
            "line_distance_weight": LINE_DISTANCE_WEIGHT,
            "top_k": TOP_K,
            "max_rows": "full" if MAX_ROWS is None else MAX_ROWS,
            "max_files_per_repo": MAX_FILES_PER_REPO,
        },
    ]

    report = []
    report.append("RepoEval full locality reranking run")
    report.append("")
    report.append(f"Total benchmark examples loaded: {total_examples}")
    report.append(f"Usable examples: {baseline_summary['usable_examples']}")
    report.append(f"Skipped (repo not found): {skipped_repo}")
    report.append(f"Skipped (repo files missing): {skipped_files}")
    report.append(f"Skipped (no chunks): {skipped_chunks}")
    report.append(f"Skipped (gold chunk not matched): {skipped_gold}")
    report.append("")
    report.append("Baseline summary:")
    report.append(str(baseline_summary))
    report.append("")
    report.append("Reranked summary:")
    report.append(str(reranked_summary))
    report.append("")
    report.append(f"Reranker parameters: same_file_bonus={SAME_FILE_BONUS}, line_distance_weight={LINE_DISTANCE_WEIGHT}")

    save_csv(PER_EXAMPLE_CSV, per_example_rows)
    save_csv(SUMMARY_CSV, summary_rows)
    save_text(REPORT_TXT, "\n".join(report))

    print(f"Saved per-example results to: {PER_EXAMPLE_CSV}")
    print(f"Saved summary results to: {SUMMARY_CSV}")
    print(f"Saved report to: {REPORT_TXT}")
    print()
    print("Baseline summary:")
    print(baseline_summary)
    print()
    print("Reranked summary:")
    print(reranked_summary)


if __name__ == "__main__":
    main()