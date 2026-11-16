"""Quantify target-file answer leakage in the pre-fix (untruncated-index) RepoEval runs.

In those runs the target file was chunked in full, so a same-file chunk whose line
span covers the completion line contains the reference line itself. This script
uses the logged top-1 chunk spans to measure how often that happened and how much
of the reported locality gain it explains.

Usage: python scripts/analyze_leakage.py  (run from the repository root)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from scipy.stats import binomtest

OUT = Path("outputs")
RUNS = {
    "StarCoder2-7B": OUT / "completion_step1_1600/repoeval_completion_step1_per_example.csv",
    "StarCoder2-3B": OUT / "completion_starcoder2_3b_1600/repoeval_completion_step1_per_example.csv",
    "CodeLlama-7B": OUT / "completion_codellama_7b_1600/completion_codellama_7b_1600_per_example.csv",
    "DeepSeek-Coder-6.7B": OUT / "completion_deepseek_6b_1600/completion_deepseek_6b_1600_per_example.csv",
    "CodeQwen1.5-7B": OUT / "completion_codeqwen_7b_1600/completion_codeqwen_7b_1600_per_example.csv",
}
DENSE = OUT / "completion_dense_starcoder2_7b/dense_completion_per_example.csv"


def covers(row: dict, arm: str) -> bool:
    """True if the arm's top-1 chunk is in the target file and spans the reference line.

    line_no is the 0-based index of the reference line; chunk spans are 1-based.
    """
    ref = int(row["line_no"]) + 1
    return (
        row[f"{arm}_top1_path"] == row["path"]
        and int(row[f"{arm}_top1_start"]) <= ref <= int(row[f"{arm}_top1_end"])
    )


def em_rate(rows, arm):
    return sum(int(r[f"em_{arm}"]) for r in rows) / len(rows) if rows else float("nan")


def mcnemar_exact(wins: int, losses: int) -> float:
    n = wins + losses
    return binomtest(wins, n, 0.5).pvalue if n else 1.0


def analyse(rows):
    n = len(rows)
    res = {"n": n}
    for arm in ("bm25", "locality"):
        cov = [r for r in rows if covers(r, arm)]
        nocov = [r for r in rows if not covers(r, arm)]
        res[arm] = {
            "em": em_rate(rows, arm),
            "top1_contains_reference": len(cov),
            "em_when_contains": em_rate(cov, arm),
            "em_when_not": em_rate(nocov, arm),
        }
    wins = [r for r in rows if int(r["em_locality"]) == 1 and int(r["em_bm25"]) == 0]
    losses = [r for r in rows if int(r["em_locality"]) == 0 and int(r["em_bm25"]) == 1]
    res["wins"] = len(wins)
    res["losses"] = len(losses)
    res["wins_where_locality_newly_exposes_reference"] = sum(
        1 for r in wins if covers(r, "locality") and not covers(r, "bm25"))
    res["losses_where_locality_hides_reference"] = sum(
        1 for r in losses if covers(r, "bm25") and not covers(r, "locality"))

    # Gain restricted to instances where neither top-1 chunk contains the reference.
    clean = [r for r in rows if not covers(r, "bm25") and not covers(r, "locality")]
    cw = sum(1 for r in clean if int(r["em_locality"]) == 1 and int(r["em_bm25"]) == 0)
    cl = sum(1 for r in clean if int(r["em_locality"]) == 0 and int(r["em_bm25"]) == 1)
    res["no_exposure_subset"] = {
        "n": len(clean),
        "em_bm25": em_rate(clean, "bm25"),
        "em_locality": em_rate(clean, "locality"),
        "wins": cw,
        "losses": cl,
        "mcnemar_p": mcnemar_exact(cw, cl),
    }
    return res


def main():
    report = {}
    for model, path in RUNS.items():
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        report[model] = analyse(rows)

    # Dense run logs only the top-1 path, so report same-file share and EM by it.
    with DENSE.open(encoding="utf-8", newline="") as f:
        drows = list(csv.DictReader(f))
    dense = {"n": len(drows)}
    for arm, col, em in (("dense", "dense_top1_path", "em_dense"),
                         ("dense_loc", "dense_loc_top1_path", "em_dense_loc")):
        same = [r for r in drows if r[col] == r["path"]]
        other = [r for r in drows if r[col] != r["path"]]
        dense[arm] = {
            "em": sum(int(r[em]) for r in drows) / len(drows),
            "top1_same_file": len(same),
            "em_when_same_file": sum(int(r[em]) for r in same) / max(1, len(same)),
            "em_when_other_file": sum(int(r[em]) for r in other) / max(1, len(other)),
        }
    report["Dense StarCoder2-7B"] = dense

    out = OUT / "leakage_analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "leakage_analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
