from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict, Any


SUMMARY_CSV = Path("outputs/tables/repoeval_locality_summary.csv")
TABLE1_OUT = Path("paper/table1_main_result.md")
TABLE2_OUT = Path("paper/table2_chunking_comparison.md")


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fmt(x: str | float) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_table1_from_summary(rows: List[Dict[str, Any]]) -> str:
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
    md.append("## Table 1. Main RepoEval retrieval result on 100 examples\n")
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
    md.append(
        f"- same_file_bonus = {rerank['same_file_bonus']}"
    )
    md.append(
        f"- line_distance_weight = {rerank['line_distance_weight']}"
    )
    md.append(
        f"- top_k = {rerank['top_k']}"
    )
    return "\n".join(md) + "\n"


def build_table2_manual() -> str:
    md = []
    md.append("## Table 2. Chunking comparison on 20 RepoEval examples\n")
    md.append("| Method | MRR | Recall@1 | Recall@5 |")
    md.append("|---|---:|---:|---:|")
    md.append("| Fixed-size chunking | 0.6014 | 0.5000 | 0.7000 |")
    md.append("| Structure-aware chunking | 0.5333 | 0.4000 | 0.7000 |")
    md.append("| Hybrid chunking | 0.4919 | 0.3500 | 0.6500 |")
    md.append("")
    md.append("Notes:")
    md.append("- Fixed chunking used chunk size = 120 and overlap = 20.")
    md.append("- Structure-aware chunking used max_chunk_tokens = 140 and overlap_tokens = 20.")
    md.append("- Hybrid chunking combined fixed and structure-aware chunks with exact-duplicate removal.")
    return "\n".join(md) + "\n"


def main():
    summary_rows = read_csv(SUMMARY_CSV)

    table1 = build_table1_from_summary(summary_rows)
    table2 = build_table2_manual()

    write_text(TABLE1_OUT, table1)
    write_text(TABLE2_OUT, table2)

    print(f"Saved: {TABLE1_OUT}")
    print(f"Saved: {TABLE2_OUT}")


if __name__ == "__main__":
    main()