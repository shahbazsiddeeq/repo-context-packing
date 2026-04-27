"""Grid search over locality reranker hyperparameters (beta, lambda).

Runs BM25 retrieval once per example, then applies all (beta, lambda)
combinations as reranking — pure arithmetic, no model inference.
Output: CSV + JSON with MRR, Recall@1, Recall@5 for every combination.
"""
from __future__ import annotations

import csv
import itertools
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

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

MAX_ROWS = int(os.getenv("MAX_ROWS", "1600"))
TOP_K = 10
MAX_FILES_PER_REPO = 1000
CHUNK_SIZE = 120
CHUNK_OVERLAP = 20

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/ablation_beta_lambda"))

# 7 x 7 = 49 combinations
BETAS   = [25.0, 50.0, 100.0, 150.0, 200.0, 300.0, 500.0]
LAMBDAS = [0.0,  0.05,  0.1,   0.2,   0.5,   1.0,   2.0]

CURRENT_BETA   = 150.0
CURRENT_LAMBDA = 0.2


def reciprocal_rank(results, gold_chunk_id: str) -> float:
    for r in results:
        if r.chunk_id == gold_chunk_id:
            return 1.0 / r.rank
    return 0.0


def recall_at_k(results, gold_chunk_id: str, k: int) -> float:
    return 1.0 if any(r.chunk_id == gold_chunk_id for r in results[:k]) else 0.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MAX_ROWS} RepoEval examples...")
    examples = load_repoeval_file(REPOEVAL_FILE, max_rows=MAX_ROWS)
    print(f"Loaded: {len(examples)} examples")

    param_grid: List[Tuple[float, float]] = list(itertools.product(BETAS, LAMBDAS))
    print(f"Grid: {len(BETAS)} beta x {len(LAMBDAS)} lambda = {len(param_grid)} combos")
    print(f"Current paper setting: beta={CURRENT_BETA}, lambda={CURRENT_LAMBDA}\n")

    # Accumulators: (beta, lambda) -> {rr, r1, r5, n}
    metrics: Dict[Tuple[float, float], Dict] = {
        p: {"rr": 0.0, "r1": 0.0, "r5": 0.0, "n": 0} for p in param_grid
    }
    baseline = {"rr": 0.0, "r1": 0.0, "r5": 0.0, "n": 0}

    skipped = 0
    for idx, ex in enumerate(examples, start=1):
        if idx % 100 == 0 or idx == 1:
            print(f"[{idx}/{len(examples)}] {ex['repo']}")

        repo_dir = find_repo_dir(ex["repo"])
        if repo_dir is None:
            skipped += 1
            continue

        repo_records = load_python_files_from_repo(
            repo_dir, max_files=MAX_FILES_PER_REPO, required_path=ex["path"]
        )
        if not repo_records:
            skipped += 1
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
            skipped += 1
            continue

        gold_id = find_gold_chunk_id(
            all_chunks, ex["path"], ex["line_no"], ex["ground_truth"]
        )
        if gold_id is None:
            skipped += 1
            continue

        # BM25 retrieval — once per example
        retriever = BM25CodeRetriever(all_chunks)
        bm25_results = retriever.search(ex["query"], top_k=TOP_K)

        # BM25 baseline
        baseline["rr"] += reciprocal_rank(bm25_results, gold_id)
        baseline["r1"] += recall_at_k(bm25_results, gold_id, 1)
        baseline["r5"] += recall_at_k(bm25_results, gold_id, 5)
        baseline["n"] += 1

        # All 49 (beta, lambda) combos — pure arithmetic on same bm25_results
        for (beta, lam) in param_grid:
            reranked = rerank_with_locality(
                bm25_results,
                current_path=ex["path"],
                current_line=ex["line_no"] + 1,
                same_file_bonus=beta,
                line_distance_weight=lam,
            )
            m = metrics[(beta, lam)]
            m["rr"] += reciprocal_rank(reranked, gold_id)
            m["r1"] += recall_at_k(reranked, gold_id, 1)
            m["r5"] += recall_at_k(reranked, gold_id, 5)
            m["n"] += 1

    usable = baseline["n"]
    print(f"\nUsable: {usable}/{len(examples)}  Skipped: {skipped}")

    # Build results rows
    rows = []
    n = baseline["n"]
    rows.append({
        "beta": 0.0, "lambda": 0.0,
        "label": "BM25-baseline",
        "MRR":      round(baseline["rr"] / n, 6),
        "Recall@1": round(baseline["r1"] / n, 6),
        "Recall@5": round(baseline["r5"] / n, 6),
        "usable": n,
        "is_paper_setting": False,
    })
    for (beta, lam) in param_grid:
        m = metrics[(beta, lam)]
        n = m["n"]
        is_paper = (beta == CURRENT_BETA and lam == CURRENT_LAMBDA)
        rows.append({
            "beta": beta, "lambda": lam,
            "label": f"b{beta:.0f}_l{lam}",
            "MRR":      round(m["rr"] / n, 6),
            "Recall@1": round(m["r1"] / n, 6),
            "Recall@5": round(m["r5"] / n, 6),
            "usable": n,
            "is_paper_setting": is_paper,
        })

    # Save CSV
    csv_path = OUTPUT_DIR / "ablation_results.csv"
    fieldnames = ["beta", "lambda", "label", "MRR", "Recall@1", "Recall@5", "usable", "is_paper_setting"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Save JSON
    json_path = OUTPUT_DIR / "ablation_results.json"
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)

    # Print table
    print("\n" + "=" * 80)
    print(f"beta/lambda Sensitivity — {usable} examples | {len(param_grid)} combos")
    print(f"{'beta':>8} {'lambda':>8} {'MRR':>10} {'R@1':>10} {'R@5':>10}")
    print("-" * 80)
    base = rows[0]
    print(f"{'BM25':>8} {'—':>8} {base['MRR']:>10.4f} {base['Recall@1']:>10.4f} {base['Recall@5']:>10.4f}  <- baseline")
    print("-" * 80)
    for row in rows[1:]:
        marker = "  <- PAPER SETTING" if row["is_paper_setting"] else ""
        print(
            f"{row['beta']:>8.0f} {row['lambda']:>8.2f} "
            f"{row['MRR']:>10.4f} {row['Recall@1']:>10.4f} {row['Recall@5']:>10.4f}{marker}"
        )
    print("=" * 80)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
