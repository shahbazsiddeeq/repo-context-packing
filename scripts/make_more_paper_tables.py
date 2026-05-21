from __future__ import annotations

from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent
PAPER_DIR  = REPO_ROOT.parent / "repo-context-packing-paper" / "paper"

TABLE3_OUT = PAPER_DIR / "table3_structure_chunk_size_sweep.md"
TABLE4_OUT = PAPER_DIR / "table4_packing_comparison.md"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_table3() -> str:
    md = []
    md.append("## Table 3. Structure-aware chunk-size sweep on RepoEval (5-example pilot)\n")
    md.append("| Max Structure Chunk Tokens | MRR | Recall@1 | Recall@5 |")
    md.append("|---:|---:|---:|---:|")
    md.append("| 60  | 0.2667 | 0.0000 | 0.6000 |")
    md.append("| 100 | 0.5000 | 0.2000 | 0.8000 |")
    md.append("| 140 | 0.9000 | 0.8000 | 1.0000 |")
    md.append("| 200 | 0.9000 | 0.8000 | 1.0000 |")
    md.append("")
    md.append("Notes:")
    md.append("- Very small structure chunks hurt retrieval substantially.")
    md.append("- Performance stabilizes around 140–200 tokens on this pilot subset.")
    md.append("- Despite this, fixed chunking remained stronger on the later 20-example comparison.")
    return "\n".join(md) + "\n"


def build_table4() -> str:
    md = []
    md.append("## Table 4. Packing comparison on fixed chunks (20 RepoEval examples)\n")
    md.append("| Budget | Method | GoldChunkPackedRate | GoldLinePackedRate | AvgPackedTokens | AvgPackedChunks |")
    md.append("|---:|---|---:|---:|---:|---:|")
    md.append("| 80  | Top-k packing | 0.0500 | 0.1000 | 8.4500 | 0.1500 |")
    md.append("| 80  | Fixed-neighbor packing | 0.0500 | 0.1000 | 8.6000 | 0.1500 |")
    md.append("| 120 | Top-k packing | 0.1000 | 0.4000 | 92.3500 | 0.8500 |")
    md.append("| 120 | Fixed-neighbor packing | 0.0500 | 0.3500 | 93.0500 | 0.8500 |")
    md.append("| 200 | Top-k packing | 0.5000 | 0.7500 | 131.2500 | 1.1500 |")
    md.append("| 200 | Fixed-neighbor packing | 0.5000 | 0.7500 | 132.2500 | 1.1500 |")
    md.append("")
    md.append("Notes:")
    md.append("- Fixed-neighbor packing did not outperform top-k packing on this subset.")
    md.append("- Retrieval quality remains the main bottleneck in the current setup.")
    md.append("- Low budgets frequently fail because the relevant chunk does not fit or is not ranked early enough.")
    return "\n".join(md) + "\n"


def main():
    write_text(TABLE3_OUT, build_table3())
    write_text(TABLE4_OUT, build_table4())

    print(f"Saved: {TABLE3_OUT}")
    print(f"Saved: {TABLE4_OUT}")


if __name__ == "__main__":
    main()