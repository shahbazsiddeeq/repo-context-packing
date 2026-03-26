from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import re

from rank_bm25 import BM25Okapi


@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    rank: int
    text: str
    metadata: Dict[str, Any]


def simple_tokenize(text: str) -> List[str]:
    """
    Very lightweight tokenizer for baseline BM25.
    Splits on words and keeps identifiers reasonably intact.
    """
    if not text:
        return []
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())


class BM25CodeRetriever:
    def __init__(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            raise ValueError("chunks must not be empty")

        self.chunks = chunks
        self.tokenized_corpus = [simple_tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        query_tokens = simple_tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results: List[RetrievalResult] = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            chunk = self.chunks[idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk["chunk_id"],
                    score=float(score),
                    rank=rank,
                    text=chunk["text"],
                    metadata={
                        "repo": chunk.get("repo"),
                        "path": chunk.get("path"),
                        "language": chunk.get("language"),
                        "start_line": chunk.get("start_line"),
                        "end_line": chunk.get("end_line"),
                    },
                )
            )

        return results


if __name__ == "__main__":
    demo_chunks = [
        {
            "chunk_id": "repo1:file1.py:chunk0",
            "repo": "repo1",
            "path": "file1.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "text": "def add(a, b):\n    return a + b",
        },
        {
            "chunk_id": "repo1:file1.py:chunk1",
            "repo": "repo1",
            "path": "file1.py",
            "language": "python",
            "start_line": 6,
            "end_line": 10,
            "text": "def subtract(a, b):\n    return a - b",
        },
        {
            "chunk_id": "repo1:file2.py:chunk0",
            "repo": "repo1",
            "path": "file2.py",
            "language": "python",
            "start_line": 1,
            "end_line": 6,
            "text": "class MathOps:\n    def multiply(self, a, b):\n        return a * b",
        },
    ]

    retriever = BM25CodeRetriever(demo_chunks)

    query = "function to add two numbers"
    results = retriever.search(query, top_k=2)

    for r in results:
        print("=" * 60)
        print(f"Rank: {r.rank}")
        print(f"Chunk ID: {r.chunk_id}")
        print(f"Score: {r.score:.4f}")
        print(f"Path: {r.metadata['path']}")
        print(r.text)