from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from src.chunking.fixed_chunker import chunk_record_fixed
from src.retrieval.bm25_retriever import BM25CodeRetriever
from src.packing.topk_packer import pack_topk_results


def load_jsonl(path: Path, max_rows: int = 5) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            rows.append(json.loads(line))
    return rows


def normalize_codesearchnet_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize CodeXGLUE/CodeSearchNet-derived record to our internal format.
    Adjust field names if needed after inspecting your local sample.
    """
    return {
        "repo": row.get("repo", "unknown_repo"),
        "path": row.get("path", "unknown_path"),
        "language": row.get("language", "python"),
        "code": row.get("code", ""),
        "query": row.get("docstring", "") or row.get("doc", "") or row.get("query", ""),
    }


def main():
    sample_path = Path("data/raw/codesearchnet/python_train_sample.jsonl")

    if not sample_path.exists():
        raise FileNotFoundError(f"Missing sample file: {sample_path}")

    raw_rows = load_jsonl(sample_path, max_rows=5)
    records = [normalize_codesearchnet_record(r) for r in raw_rows]

    all_chunks = []
    usable_records = []

    for record in records:
        if not record["code"].strip():
            continue
        if not record["query"].strip():
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

        if not chunks:
            continue

        all_chunks.extend(chunks)
        usable_records.append(record)

    if not all_chunks:
        raise RuntimeError("No chunks created from sample data.")

    print("=" * 100)
    print(f"Loaded usable records: {len(usable_records)}")
    print(f"Total chunks created: {len(all_chunks)}")

    retriever = BM25CodeRetriever(all_chunks)

    query = usable_records[0]["query"]
    print(f"\nSmoke-test query:\n{query}")

    results = retriever.search(query, top_k=5)
    packed = pack_topk_results(results, max_context_tokens=200)

    print("\nTop retrieval results:")
    for r in results:
        print("-" * 80)
        print(f"rank={r.rank} score={r.score:.4f}")
        print(f"chunk_id={r.chunk_id}")
        print(f"path={r.metadata.get('path')}")
        print(r.text[:300])

    print("\n" + "=" * 100)
    print("Packed context summary")
    print(f"num_chunks={packed.num_chunks}")
    print(f"total_tokens={packed.total_tokens}")
    print("-" * 80)
    print(packed.context_text[:1200])


if __name__ == "__main__":
    main()