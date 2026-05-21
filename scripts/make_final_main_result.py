from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict, Any

REPO_ROOT      = Path(__file__).resolve().parent.parent
PAPER_DIR      = REPO_ROOT.parent / "repo-context-packing-paper" / "paper"

FULL_SUMMARY_CSV = REPO_ROOT / "outputs" / "full_run" / "repoeval_full_locality_summary.csv"
ANALYSIS_TXT     = REPO_ROOT / "outputs" / "analysis" / "locality_significance_summary.txt"

TABLE1_OUT     = PAPER_DIR / "table1_main_result.md"
HIGHLIGHTS_OUT = PAPER_DIR / "result_highlights_v1.md"


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fmt(x: Any) -> str:
    return f"{float(x):.4f}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_table1(rows: List[Dict[str, Any]]) -> str:
    fixed = None
    rerank = None

    for row in rows:
        if row["method"] == "fixed_bm25":
            fixed = row
        elif row["method"] == "fixed_bm25_locality_rerank":
            rerank = row

    if fixed is None or rerank is None:
        raise RuntimeError("Could not find fixed_bm25 and fixed_bm25_locality_rerank rows.")

    md = []
    md.append("## Table 1. Main RepoEval retrieval result on the full 1600-example line-level split\n")
    md.append("| Method | Usable Examples | MRR | Recall@1 | Recall@5 |")
    md.append("|---|---:|---:|---:|---:|")
    md.append(
        f"| Fixed BM25 | {fixed['usable_examples']} | {fmt(fixed['MRR'])} | {fmt(fixed['Recall@1'])} | {fmt(fixed['Recall@5'])} |"
    )
    md.append(
        f"| Fixed BM25 + Locality Reranking | {rerank['usable_examples']} | {fmt(rerank['MRR'])} | {fmt(rerank['Recall@1'])} | {fmt(rerank['Recall@5'])} |"
    )
    md.append("")
    md.append("Best reranking setting used for this table:")
    md.append(f"- same_file_bonus = {rerank['same_file_bonus']}")
    md.append(f"- line_distance_weight = {rerank['line_distance_weight']}")
    md.append(f"- top_k = {rerank['top_k']}")
    md.append(f"- max_files_per_repo = {rerank['max_files_per_repo']}")
    return "\n".join(md) + "\n"


def build_highlights(rows: List[Dict[str, Any]], analysis_text: str) -> str:
    fixed = None
    rerank = None

    for row in rows:
        if row["method"] == "fixed_bm25":
            fixed = row
        elif row["method"] == "fixed_bm25_locality_rerank":
            rerank = row

    if fixed is None or rerank is None:
        raise RuntimeError("Could not find fixed_bm25 and fixed_bm25_locality_rerank rows.")

    delta_mrr = float(rerank["MRR"]) - float(fixed["MRR"])
    delta_r1 = float(rerank["Recall@1"]) - float(fixed["Recall@1"])
    delta_r5 = float(rerank["Recall@5"]) - float(fixed["Recall@5"])

    md = []
    md.append("# Main Result Highlights\n")
    md.append("## Full-run result")
    md.append(
        f"- Fixed BM25: MRR {fmt(fixed['MRR'])}, Recall@1 {fmt(fixed['Recall@1'])}, Recall@5 {fmt(fixed['Recall@5'])}"
    )
    md.append(
        f"- Fixed BM25 + locality reranking: MRR {fmt(rerank['MRR'])}, Recall@1 {fmt(rerank['Recall@1'])}, Recall@5 {fmt(rerank['Recall@5'])}"
    )
    md.append("")
    md.append("## Absolute gains")
    md.append(f"- Delta MRR: {delta_mrr:.4f}")
    md.append(f"- Delta Recall@1: {delta_r1:.4f}")
    md.append(f"- Delta Recall@5: {delta_r5:.4f}")
    md.append("")
    md.append("## Significance / robustness summary")
    md.append("Copy the following directly into Results or Discussion after checking wording:")
    md.append("")
    md.append("> Locality-aware reranking improved MRR by 0.0334 with a 95% bootstrap confidence interval of [0.0267, 0.0404]. It improved Recall@1 by 0.0375 with a 95% confidence interval of [0.0275, 0.0481], and Recall@5 by 0.0275 with a 95% confidence interval of [0.0181, 0.0375]. All intervals remained above zero on the evaluated split.")
    md.append("")
    md.append("> Error analysis showed that reranking improved the gold-chunk rank in 186 cases while worsening it in only 40 cases. At the top rank, reranking produced 67 Recall@1 gains and only 7 Recall@1 losses. In 93 cases, the top-ranked result moved from a different file to the correct same file after reranking.")
    md.append("")
    md.append("## Source note")
    md.append("- Metrics source: outputs/full_run/repoeval_full_locality_summary.csv")
    md.append("- Analysis source: outputs/analysis/locality_significance_summary.txt")
    return "\n".join(md) + "\n"


def main():
    rows = read_csv(FULL_SUMMARY_CSV)
    analysis_text = read_text(ANALYSIS_TXT)

    table1 = build_table1(rows)
    highlights = build_highlights(rows, analysis_text)

    write_text(TABLE1_OUT, table1)
    write_text(HIGHLIGHTS_OUT, highlights)

    print(f"Saved: {TABLE1_OUT}")
    print(f"Saved: {HIGHLIGHTS_OUT}")


if __name__ == "__main__":
    main()