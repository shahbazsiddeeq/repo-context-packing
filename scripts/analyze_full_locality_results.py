from __future__ import annotations

import csv
import random
from pathlib import Path
from statistics import mean
from typing import Dict, Any, List


INPUT_CSV = Path("outputs/full_run/repoeval_full_locality_per_example.csv")
OUTPUT_DIR = Path("outputs/analysis")
SUMMARY_TXT = OUTPUT_DIR / "locality_significance_summary.txt"
ERROR_BUCKETS_CSV = OUTPUT_DIR / "locality_error_buckets.csv"

BOOTSTRAP_SAMPLES = 10000
RANDOM_SEED = 42


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(x: Any) -> float:
    return float(x)


def to_int(x: Any) -> int:
    return int(float(x))


def bootstrap_ci(deltas: List[float], samples: int = BOOTSTRAP_SAMPLES, seed: int = RANDOM_SEED):
    rng = random.Random(seed)
    n = len(deltas)
    means = []

    for _ in range(samples):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))

    means.sort()
    lo = means[int(0.025 * samples)]
    hi = means[int(0.975 * samples)]
    return lo, hi


def paired_sign_test(deltas: List[float]) -> Dict[str, Any]:
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    zero = sum(1 for d in deltas if d == 0)
    return {"positive": pos, "negative": neg, "zero": zero}


def classify_row(row: Dict[str, Any]) -> Dict[str, Any]:
    baseline_rank = to_int(row["baseline_gold_rank"])
    rerank_rank = to_int(row["rerank_gold_rank"])

    baseline_top1_path = row["baseline_top1_path"]
    rerank_top1_path = row["rerank_top1_path"]
    target_path = row["path"]

    baseline_r1 = to_float(row["baseline_r1"])
    rerank_r1 = to_float(row["rerank_r1"])
    baseline_r5 = to_float(row["baseline_r5"])
    rerank_r5 = to_float(row["rerank_r5"])

    if rerank_rank == -1 and baseline_rank == -1:
        movement = "still_missed"
    elif baseline_rank == -1 and rerank_rank != -1:
        movement = "recovered_from_miss"
    elif baseline_rank != -1 and rerank_rank == -1:
        movement = "lost_after_rerank"
    elif rerank_rank < baseline_rank:
        movement = "improved_rank"
    elif rerank_rank > baseline_rank:
        movement = "worsened_rank"
    else:
        movement = "same_rank"

    if baseline_top1_path == target_path and rerank_top1_path == target_path:
        path_transition = "same_file_to_same_file"
    elif baseline_top1_path != target_path and rerank_top1_path == target_path:
        path_transition = "other_file_to_same_file"
    elif baseline_top1_path == target_path and rerank_top1_path != target_path:
        path_transition = "same_file_to_other_file"
    else:
        path_transition = "other_file_to_other_file"

    if baseline_r1 == 0.0 and rerank_r1 == 1.0:
        top1_transition = "r1_gain"
    elif baseline_r1 == 1.0 and rerank_r1 == 0.0:
        top1_transition = "r1_loss"
    else:
        top1_transition = "r1_same"

    if baseline_r5 == 0.0 and rerank_r5 == 1.0:
        top5_transition = "r5_gain"
    elif baseline_r5 == 1.0 and rerank_r5 == 0.0:
        top5_transition = "r5_loss"
    else:
        top5_transition = "r5_same"

    return {
        "task_id": row["task_id"],
        "repo": row["repo"],
        "path": target_path,
        "line_no": row["line_no"],
        "baseline_gold_rank": baseline_rank,
        "rerank_gold_rank": rerank_rank,
        "baseline_rr": row["baseline_rr"],
        "rerank_rr": row["rerank_rr"],
        "baseline_r1": row["baseline_r1"],
        "rerank_r1": row["rerank_r1"],
        "baseline_r5": row["baseline_r5"],
        "rerank_r5": row["rerank_r5"],
        "baseline_top1_path": baseline_top1_path,
        "rerank_top1_path": rerank_top1_path,
        "movement": movement,
        "path_transition": path_transition,
        "top1_transition": top1_transition,
        "top5_transition": top5_transition,
    }


def count_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        val = row[key]
        out[val] = out.get(val, 0) + 1
    return dict(sorted(out.items(), key=lambda x: (-x[1], x[0])))


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to save for {path}")

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    rows = read_csv(INPUT_CSV)
    n = len(rows)
    if n == 0:
        raise RuntimeError("No rows found in per-example CSV.")

    baseline_rr = [to_float(r["baseline_rr"]) for r in rows]
    rerank_rr = [to_float(r["rerank_rr"]) for r in rows]
    baseline_r1 = [to_float(r["baseline_r1"]) for r in rows]
    rerank_r1 = [to_float(r["rerank_r1"]) for r in rows]
    baseline_r5 = [to_float(r["baseline_r5"]) for r in rows]
    rerank_r5 = [to_float(r["rerank_r5"]) for r in rows]

    rr_delta = [b - a for a, b in zip(baseline_rr, rerank_rr)]
    r1_delta = [b - a for a, b in zip(baseline_r1, rerank_r1)]
    r5_delta = [b - a for a, b in zip(baseline_r5, rerank_r5)]

    rr_ci = bootstrap_ci(rr_delta)
    r1_ci = bootstrap_ci(r1_delta)
    r5_ci = bootstrap_ci(r5_delta)

    rr_sign = paired_sign_test(rr_delta)
    r1_sign = paired_sign_test(r1_delta)
    r5_sign = paired_sign_test(r5_delta)

    classified = [classify_row(r) for r in rows]
    movement_counts = count_by(classified, "movement")
    path_counts = count_by(classified, "path_transition")
    top1_counts = count_by(classified, "top1_transition")
    top5_counts = count_by(classified, "top5_transition")

    save_csv(ERROR_BUCKETS_CSV, classified)

    report = []
    report.append("Locality reranking significance and error analysis")
    report.append("")
    report.append(f"Number of examples analyzed: {n}")
    report.append("")
    report.append("Aggregate metrics")
    report.append(f"- Baseline MRR: {mean(baseline_rr):.4f}")
    report.append(f"- Reranked MRR: {mean(rerank_rr):.4f}")
    report.append(f"- Delta MRR: {mean(rr_delta):.4f}")
    report.append(f"- Delta MRR 95% bootstrap CI: [{rr_ci[0]:.4f}, {rr_ci[1]:.4f}]")
    report.append(f"- Delta MRR sign counts: {rr_sign}")
    report.append("")
    report.append(f"- Baseline Recall@1: {mean(baseline_r1):.4f}")
    report.append(f"- Reranked Recall@1: {mean(rerank_r1):.4f}")
    report.append(f"- Delta Recall@1: {mean(r1_delta):.4f}")
    report.append(f"- Delta Recall@1 95% bootstrap CI: [{r1_ci[0]:.4f}, {r1_ci[1]:.4f}]")
    report.append(f"- Delta Recall@1 sign counts: {r1_sign}")
    report.append("")
    report.append(f"- Baseline Recall@5: {mean(baseline_r5):.4f}")
    report.append(f"- Reranked Recall@5: {mean(rerank_r5):.4f}")
    report.append(f"- Delta Recall@5: {mean(r5_delta):.4f}")
    report.append(f"- Delta Recall@5 95% bootstrap CI: [{r5_ci[0]:.4f}, {r5_ci[1]:.4f}]")
    report.append(f"- Delta Recall@5 sign counts: {r5_sign}")
    report.append("")
    report.append("Error analysis buckets")
    report.append(f"- Movement counts: {movement_counts}")
    report.append(f"- Path transition counts: {path_counts}")
    report.append(f"- Top1 transition counts: {top1_counts}")
    report.append(f"- Top5 transition counts: {top5_counts}")
    report.append("")
    report.append(f"Detailed per-example buckets saved to: {ERROR_BUCKETS_CSV}")

    save_text(SUMMARY_TXT, "\n".join(report))

    print(f"Saved summary report to: {SUMMARY_TXT}")
    print(f"Saved error buckets CSV to: {ERROR_BUCKETS_CSV}")
    print()
    print("\n".join(report))


if __name__ == "__main__":
    main()