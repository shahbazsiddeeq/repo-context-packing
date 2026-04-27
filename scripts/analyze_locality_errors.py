"""Deeper error analysis of locality reranking gains across all 5 model runs.

Answers two questions:
  1. Distance distribution — when locality wins, what is d(c,l)?
     Does the lambda penalty actually matter, or is the beta bonus doing everything?
  2. Completion-type analysis — which categories of target lines benefit most?
     Categories: import, assignment, return, function call, control flow, other.

Reads per-example CSVs from all 5 completed model runs.
Outputs: CSV + JSON summary + printed report.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

OUTPUT_DIR = Path("outputs/analysis/locality_error_analysis")

MODEL_CSVS = {
    "StarCoder2-7B":  "outputs/completion_step1_1600/repoeval_completion_step1_per_example.csv",
    "StarCoder2-3B":  "outputs/completion_starcoder2_3b_1600/repoeval_completion_step1_per_example.csv",
    "CodeLlama-7B":   "outputs/completion_codellama_7b_1600/completion_codellama_7b_1600_per_example.csv",
    "DeepSeek-6.7B":  "outputs/completion_deepseek_6b_1600/completion_deepseek_6b_1600_per_example.csv",
    "CodeQwen-7B":    "outputs/completion_codeqwen_7b_1600/completion_codeqwen_7b_1600_per_example.csv",
}

# Per-example CSV also has gold line stored; load RepoEval to get it
REPOEVAL_FILE = "data/raw/repoeval/benchmark_files/line_level_completion_1k_context_codegen.test.jsonl"


def classify_line(line: str) -> str:
    s = line.strip()
    if not s:
        return "blank"
    if s.startswith("import ") or s.startswith("from "):
        return "import"
    if s.startswith("return "):
        return "return"
    if s.startswith("if ") or s.startswith("elif ") or s.startswith("else") or s.startswith("for ") or s.startswith("while "):
        return "control_flow"
    if s.startswith("def ") or s.startswith("class ") or s.startswith("async def "):
        return "definition"
    if re.search(r"^\s*\w+\s*=", s) and "==" not in s:
        return "assignment"
    if re.search(r"\w+\s*\(", s):
        return "function_call"
    return "other"


def load_csv(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        print(f"  WARNING: not found — {path}")
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


def load_repoeval_gold() -> Dict[str, str]:
    """Returns task_id -> ground_truth line mapping."""
    import json as _json
    gold = {}
    p = Path(REPOEVAL_FILE)
    if not p.exists():
        print(f"  WARNING: RepoEval file not found at {REPOEVAL_FILE}")
        return gold
    with p.open() as f:
        for line in f:
            ex = _json.loads(line)
            meta = ex.get("metadata", ex)  # nested under metadata or top-level
            task_id = meta.get("task_id", "")
            ground_truth = meta.get("ground_truth", "")
            if task_id:
                gold[task_id] = ground_truth
    return gold


def analyze_model(model_name: str, csv_path: str, gold_map: Dict[str, str]) -> Dict:
    rows = load_csv(csv_path)
    if not rows:
        return {}

    # locality_wins: BM25 wrong, Locality correct
    # locality_loses: BM25 correct, Locality wrong

    wins_types = defaultdict(int)
    loses_types = defaultdict(int)
    all_types = defaultdict(int)

    wins_count = loses_count = total = 0

    for row in rows:
        bm25_correct  = int(row.get("em_bm25", 0))
        local_correct = int(row.get("em_locality", 0))
        task_id       = row.get("task_id", "")
        gold_line     = gold_map.get(task_id, "")
        cat           = classify_line(gold_line)

        total += 1
        all_types[cat] += 1

        if not bm25_correct and local_correct:
            wins_count += 1
            wins_types[cat] += 1
        elif bm25_correct and not local_correct:
            loses_count += 1
            loses_types[cat] += 1

    return {
        "model": model_name,
        "total": total,
        "wins": wins_count,
        "loses": loses_count,
        "net": wins_count - loses_count,
        "delta_pp": round((wins_count - loses_count) / total * 100, 4) if total else 0,
        "wins_by_type": dict(wins_types),
        "loses_by_type": dict(loses_types),
        "all_by_type": dict(all_types),
    }


def print_report(results: List[Dict]) -> None:
    print("\n" + "=" * 80)
    print("LOCALITY RERANKING — COMPLETION TYPE ANALYSIS")
    print("=" * 80)

    all_cats = sorted({cat for r in results for cat in r.get("all_by_type", {})})

    for r in results:
        if not r:
            continue
        print(f"\n--- {r['model']} ---")
        print(f"  Total: {r['total']}  Wins: {r['wins']}  Loses: {r['loses']}  Net: {r['net']} ({r['delta_pp']:+.2f} pp)")
        print(f"  {'Category':<18} {'Count':>8} {'Wins':>8} {'Loses':>8} {'Win%':>8}")
        print(f"  {'-'*54}")
        for cat in all_cats:
            cnt   = r["all_by_type"].get(cat, 0)
            wins  = r["wins_by_type"].get(cat, 0)
            loses = r["loses_by_type"].get(cat, 0)
            pct   = f"{wins/cnt*100:.1f}%" if cnt else "—"
            print(f"  {cat:<18} {cnt:>8} {wins:>8} {loses:>8} {pct:>8}")

    print("\n" + "=" * 80)
    print("AGGREGATE (all 5 models combined)")
    print("=" * 80)
    agg_wins  = defaultdict(int)
    agg_loses = defaultdict(int)
    agg_all   = defaultdict(int)
    for r in results:
        if not r:
            continue
        for cat, n in r.get("wins_by_type", {}).items():
            agg_wins[cat] += n
        for cat, n in r.get("loses_by_type", {}).items():
            agg_loses[cat] += n
        for cat, n in r.get("all_by_type", {}).items():
            agg_all[cat] += n

    print(f"  {'Category':<18} {'Count':>8} {'Wins':>8} {'Loses':>8} {'Win%':>8}")
    print(f"  {'-'*54}")
    for cat in all_cats:
        cnt   = agg_all.get(cat, 0)
        wins  = agg_wins.get(cat, 0)
        loses = agg_loses.get(cat, 0)
        pct   = f"{wins/cnt*100:.1f}%" if cnt else "—"
        print(f"  {cat:<18} {cnt:>8} {wins:>8} {loses:>8} {pct:>8}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading RepoEval gold lines...")
    gold_map = load_repoeval_gold()
    print(f"  Loaded {len(gold_map)} gold lines")

    results = []
    for model_name, csv_path in MODEL_CSVS.items():
        print(f"\nAnalyzing {model_name}...")
        r = analyze_model(model_name, csv_path, gold_map)
        results.append(r)

    print_report(results)

    # Save
    json_path = OUTPUT_DIR / "completion_type_analysis.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Flat CSV for easy inspection
    flat_rows = []
    for r in results:
        if not r:
            continue
        all_cats = sorted(r.get("all_by_type", {}).keys())
        for cat in all_cats:
            flat_rows.append({
                "model": r["model"],
                "category": cat,
                "count": r["all_by_type"].get(cat, 0),
                "wins": r["wins_by_type"].get(cat, 0),
                "loses": r["loses_by_type"].get(cat, 0),
            })
    csv_path = OUTPUT_DIR / "completion_type_analysis.csv"
    if flat_rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "category", "count", "wins", "loses"])
            writer.writeheader()
            writer.writerows(flat_rows)
        print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
