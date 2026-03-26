from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

from src.chunking.fixed_chunker import chunk_record_fixed
from src.chunking.structure_chunker import chunk_record_structure_aware
from src.retrieval.bm25_retriever import BM25CodeRetriever
from src.packing.topk_packer import pack_topk_results
from src.packing.dependency_packer import pack_dependency_aware_results


def load_jsonl(path: Path, max_rows: int = 20) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            rows.append(json.loads(line))
    return rows


def normalize_codesearchnet_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "repo": row.get("repo", "unknown_repo"),
        "path": row.get("path", "unknown_path"),
        "language": row.get("language", "python"),
        "code": row.get("code", ""),
        "query": row.get("docstring", "") or row.get("doc", "") or row.get("query", ""),
    }


def reciprocal_rank(results, gold_chunk_id: str) -> float:
    for r in results:
        if r.chunk_id == gold_chunk_id:
            return 1.0 / r.rank
    return 0.0


def recall_at_k(results, gold_chunk_id: str, k: int) -> float:
    return 1.0 if any(r.chunk_id == gold_chunk_id for r in results[:k]) else 0.0


def find_gold_chunk_id(chunks: List[Dict[str, Any]], gold_code_text: str) -> str | None:
    gold_code_text = gold_code_text.strip()
    for chunk in chunks:
        if gold_code_text and gold_code_text in chunk.get("text", ""):
            return chunk["chunk_id"]
    return None


def build_chunk_corpus(
    records: List[Dict[str, Any]],
    mode: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      all_chunks, usable_records_with_gold
    """
    all_chunks: List[Dict[str, Any]] = []
    usable_records: List[Dict[str, Any]] = []

    for record in records:
        if not record["code"].strip() or not record["query"].strip():
            continue

        if mode == "fixed":
            chunks = chunk_record_fixed(
                record,
                code_field="code",
                repo_field="repo",
                path_field="path",
                language_field="language",
                chunk_size_tokens=120,
                chunk_overlap_tokens=20,
            )
        elif mode == "structure":
            chunks = chunk_record_structure_aware(
                record,
                code_field="code",
                repo_field="repo",
                path_field="path",
                language_field="language",
                fallback_to_fixed=True,
                fixed_chunk_size_tokens=120,
                fixed_chunk_overlap_tokens=20,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if not chunks:
            continue

        gold_chunk_id = find_gold_chunk_id(chunks, record["code"])
        if gold_chunk_id is None:
            continue

        all_chunks.extend(chunks)

        enriched = dict(record)
        enriched["gold_chunk_id"] = gold_chunk_id
        usable_records.append(enriched)

    return all_chunks, usable_records


def evaluate_retrieval_and_packing(
    all_chunks: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    use_dependency_packing: bool,
    top_k: int = 5,
    max_context_tokens: int = 250,
) -> Dict[str, float]:
    retriever = BM25CodeRetriever(all_chunks)

    rr_scores = []
    r1_scores = []
    r3_scores = []
    packed_token_counts = []
    packed_chunk_counts = []

    for record in records:
        query = record["query"]
        gold_chunk_id = record["gold_chunk_id"]

        results = retriever.search(query, top_k=top_k)

        rr_scores.append(reciprocal_rank(results, gold_chunk_id))
        r1_scores.append(recall_at_k(results, gold_chunk_id, 1))
        r3_scores.append(recall_at_k(results, gold_chunk_id, 3))

        if use_dependency_packing:
            packed = pack_dependency_aware_results(
                retrieval_results=results,
                all_chunks=all_chunks,
                max_context_tokens=max_context_tokens,
                max_dependency_expansions_per_chunk=2,
            )
        else:
            packed = pack_topk_results(
                retrieval_results=results,
                max_context_tokens=max_context_tokens,
            )

        packed_token_counts.append(packed.total_tokens)
        packed_chunk_counts.append(packed.num_chunks)

    n = len(records)
    if n == 0:
        raise RuntimeError("No usable records available for evaluation.")

    return {
        "num_queries": n,
        "MRR": sum(rr_scores) / n,
        "Recall@1": sum(r1_scores) / n,
        "Recall@3": sum(r3_scores) / n,
        "AvgPackedTokens": sum(packed_token_counts) / n,
        "AvgPackedChunks": sum(packed_chunk_counts) / n,
    }


def print_summary(title: str, metrics: Dict[str, float]) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("-" * 100)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:16s}: {v:.4f}")
        else:
            print(f"{k:16s}: {v}")


def main():
    sample_path = Path("data/raw/codesearchnet/python_train_sample.jsonl")
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing sample file: {sample_path}")

    raw_rows = load_jsonl(sample_path, max_rows=20)
    records = [normalize_codesearchnet_record(r) for r in raw_rows]

    fixed_chunks, fixed_records = build_chunk_corpus(records, mode="fixed")
    struct_chunks, struct_records = build_chunk_corpus(records, mode="structure")

    print(f"Fixed corpus chunks: {len(fixed_chunks)}")
    print(f"Fixed usable queries: {len(fixed_records)}")
    print(f"Structure corpus chunks: {len(struct_chunks)}")
    print(f"Structure usable queries: {len(struct_records)}")

    fixed_topk = evaluate_retrieval_and_packing(
        all_chunks=fixed_chunks,
        records=fixed_records,
        use_dependency_packing=False,
    )

    struct_topk = evaluate_retrieval_and_packing(
        all_chunks=struct_chunks,
        records=struct_records,
        use_dependency_packing=False,
    )

    struct_dep = evaluate_retrieval_and_packing(
        all_chunks=struct_chunks,
        records=struct_records,
        use_dependency_packing=True,
    )

    print_summary("Fixed-size chunking + BM25 + Top-k packing", fixed_topk)
    print_summary("Structure-aware chunking + BM25 + Top-k packing", struct_topk)
    print_summary("Structure-aware chunking + BM25 + Dependency-aware packing", struct_dep)


if __name__ == "__main__":
    main()