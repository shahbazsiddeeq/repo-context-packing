from __future__ import annotations

from typing import List, Dict, Any

from src.chunking.fixed_chunker import chunk_record_fixed
from src.retrieval.bm25_retriever import BM25CodeRetriever
from src.packing.topk_packer import pack_topk_results


def reciprocal_rank(results: List[Any], gold_chunk_id: str) -> float:
    for r in results:
        if r.chunk_id == gold_chunk_id:
            return 1.0 / r.rank
    return 0.0


def recall_at_k(results: List[Any], gold_chunk_id: str, k: int) -> float:
    top_k = results[:k]
    return 1.0 if any(r.chunk_id == gold_chunk_id for r in top_k) else 0.0


def build_demo_dataset() -> List[Dict[str, Any]]:
    return [
        {
            "repo": "demo_repo",
            "path": "math_utils.py",
            "language": "python",
            "code": "\n".join([
                "def add(a, b):",
                "    return a + b",
                "",
                "def subtract(a, b):",
                "    return a - b",
                "",
                "class MathOps:",
                "    def multiply(self, a, b):",
                "        return a * b",
            ]),
            "query": "function to add two numbers",
            "gold_match_text": "def add(a, b):",
        },
        {
            "repo": "demo_repo",
            "path": "string_utils.py",
            "language": "python",
            "code": "\n".join([
                "def to_upper(text):",
                "    return text.upper()",
                "",
                "def to_lower(text):",
                "    return text.lower()",
            ]),
            "query": "convert string to uppercase",
            "gold_match_text": "def to_upper(text):",
        },
    ]


def find_gold_chunk_id(chunks: List[Dict[str, Any]], gold_match_text: str) -> str:
    for chunk in chunks:
        if gold_match_text in chunk["text"]:
            return chunk["chunk_id"]
    raise ValueError(f"Could not find gold chunk for text: {gold_match_text}")


def evaluate_demo(
    chunk_size_tokens: int = 20,
    chunk_overlap_tokens: int = 5,
    top_k: int = 5,
    max_context_tokens: int = 80,
) -> None:
    dataset = build_demo_dataset()

    all_chunks: List[Dict[str, Any]] = []
    gold_by_query: Dict[str, str] = {}

    for record in dataset:
        chunks = chunk_record_fixed(
            record,
            code_field="code",
            repo_field="repo",
            path_field="path",
            language_field="language",
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )
        all_chunks.extend(chunks)

        gold_chunk_id = find_gold_chunk_id(chunks, record["gold_match_text"])
        gold_by_query[record["query"]] = gold_chunk_id

    retriever = BM25CodeRetriever(all_chunks)

    rr_scores = []
    recall1_scores = []
    recall3_scores = []

    for record in dataset:
        query = record["query"]
        gold_chunk_id = gold_by_query[query]

        results = retriever.search(query, top_k=top_k)

        rr = reciprocal_rank(results, gold_chunk_id)
        r1 = recall_at_k(results, gold_chunk_id, k=1)
        r3 = recall_at_k(results, gold_chunk_id, k=3)

        rr_scores.append(rr)
        recall1_scores.append(r1)
        recall3_scores.append(r3)

        packed = pack_topk_results(results, max_context_tokens=max_context_tokens)

        print("=" * 100)
        print(f"Query: {query}")
        print(f"Gold chunk: {gold_chunk_id}")
        print(f"Reciprocal Rank: {rr:.4f}")
        print(f"Recall@1: {r1:.1f}")
        print(f"Recall@3: {r3:.1f}")
        print(f"Packed chunks: {packed.num_chunks}")
        print(f"Packed tokens: {packed.total_tokens}")
        print("-" * 100)
        for r in results:
            marker = " <-- GOLD" if r.chunk_id == gold_chunk_id else ""
            print(f"rank={r.rank} score={r.score:.4f} chunk_id={r.chunk_id}{marker}")

    mean_rr = sum(rr_scores) / len(rr_scores)
    mean_r1 = sum(recall1_scores) / len(recall1_scores)
    mean_r3 = sum(recall3_scores) / len(recall3_scores)

    print("\n" + "#" * 100)
    print("Aggregate metrics")
    print(f"MRR: {mean_rr:.4f}")
    print(f"Recall@1: {mean_r1:.4f}")
    print(f"Recall@3: {mean_r3:.4f}")


if __name__ == "__main__":
    evaluate_demo()