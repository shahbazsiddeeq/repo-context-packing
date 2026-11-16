"""Paired statistics for the truncated-index (clean) RepoEval run with control arms.

Usage: python scripts/analyze_clean_run.py <per_example.csv>  (from the repository root)
Writes <csv_dir>/clean_run_stats.json and prints a readable summary.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

from scipy.stats import binomtest

ARMS = ["prompt_only", "bm25", "locality", "nearest", "crossfile"]
SEED = 7
N_BOOT = 10_000


def paired(rows, a, b):
    """Compare arm a against arm b: delta in pp, bootstrap CI, gains/losses, exact McNemar p."""
    xa = [int(r[f"em_{a}"]) for r in rows]
    xb = [int(r[f"em_{b}"]) for r in rows]
    n = len(rows)
    diffs = [p - q for p, q in zip(xa, xb)]
    gains = sum(1 for d in diffs if d == 1)
    losses = sum(1 for d in diffs if d == -1)
    rng = random.Random(SEED)
    boots = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT)
    ) if n else [0.0]
    lo, hi = boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]
    p = binomtest(gains, gains + losses, 0.5).pvalue if gains + losses else 1.0
    return {
        "n": n,
        "em_a": 100 * sum(xa) / n if n else float("nan"),
        "em_b": 100 * sum(xb) / n if n else float("nan"),
        "delta_pp": 100 * sum(diffs) / n if n else float("nan"),
        "ci_pp": [100 * lo, 100 * hi],
        "gains": gains,
        "losses": losses,
        "mcnemar_p": p,
    }


def main(path: str):
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    n = len(rows)
    out = {"n": n, "em": {a: 100 * sum(int(r[f"em_{a}"]) for r in rows) / n for a in ARMS}}

    comps = [("bm25", "prompt_only"), ("locality", "prompt_only"), ("nearest", "prompt_only"),
             ("crossfile", "prompt_only"), ("locality", "bm25"), ("locality", "nearest"),
             ("crossfile", "bm25")]
    out["paired"] = {f"{a}_vs_{b}": paired(rows, a, b) for a, b in comps}

    # Sanity: no prompt chunk may span the reference line under truncation.
    def spans_ref(r, arm):
        ref = int(r["line_no"]) + 1
        return r[f"{arm}_top1_path"] == r["path"] and int(r[f"{arm}_top1_start"]) <= ref <= int(r[f"{arm}_top1_end"])
    out["exposed_bm25"] = sum(spans_ref(r, "bm25") for r in rows)
    out["exposed_locality"] = sum(spans_ref(r, "locality") for r in rows)

    # How often reranking reproduces the no-retriever nearest window.
    out["locality_equals_nearest"] = sum(
        1 for r in rows if r["locality_top1_path"] == r["path"]
        and r["locality_top1_start"] == r["nearest_start"] and r["locality_top1_end"] == r["nearest_end"])
    out["bm25_equals_nearest"] = sum(
        1 for r in rows if r["bm25_top1_path"] == r["path"]
        and r["bm25_top1_start"] == r["nearest_start"] and r["bm25_top1_end"] == r["nearest_end"])
    out["crossfile_found"] = sum(int(r["crossfile_found"]) for r in rows)

    # Header-copy artifact check (the failure of the earlier uncommented format).
    out["pred_is_comment_line"] = {
        a: sum(1 for r in rows if r[f"pred_{a}"].strip().startswith("#") and not r["gold_line"].strip().startswith("#"))
        for a in ARMS}

    # Strata by where the BM25 top-1 chunk comes from.
    same = [r for r in rows if r["bm25_top1_path"] == r["path"]]
    cross = [r for r in rows if r["bm25_top1_path"] != r["path"]]
    out["strata"] = {}
    for name, sub in (("bm25_top1_same_file", same), ("bm25_top1_cross_file", cross)):
        out["strata"][name] = {
            "n": len(sub),
            "em": {a: (100 * sum(int(r[f"em_{a}"]) for r in sub) / len(sub)) if sub else None for a in ARMS},
            "locality_vs_bm25": paired(sub, "locality", "bm25") if sub else None,
            "bm25_vs_prompt_only": paired(sub, "bm25", "prompt_only") if sub else None,
            "crossfile_vs_prompt_only": paired(sub, "crossfile", "prompt_only") if sub else None,
        }
    # Per repository.
    out["per_repo"] = {}
    for repo in sorted({r["repo"] for r in rows}):
        sub = [r for r in rows if r["repo"] == repo]
        out["per_repo"][repo] = {a: 100 * sum(int(r[f"em_{a}"]) for r in sub) / len(sub) for a in ARMS}

    dst = Path(path).parent / "clean_run_stats.json"
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
