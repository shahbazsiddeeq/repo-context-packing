from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any


def approx_token_count(text: str) -> int:
    """
    Very simple token approximation for baseline packing.
    """
    if not text:
        return 0
    return len(text.split())


@dataclass
class PackedChunk:
    chunk_id: str
    rank: int
    score: float
    path: str
    start_line: int
    end_line: int
    token_count: int
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PackedContext:
    total_tokens: int
    max_tokens: int
    num_chunks: int
    context_text: str
    packed_chunks: List[PackedChunk]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "num_chunks": self.num_chunks,
            "context_text": self.context_text,
            "packed_chunks": [c.to_dict() for c in self.packed_chunks],
        }


def pack_topk_results(
    retrieval_results: List[Any],
    max_context_tokens: int = 400,
    separator: str = "\n\n" + "=" * 80 + "\n\n",
) -> PackedContext:
    """
    Pack retrieved chunks in rank order until the token budget is exhausted.

    Parameters
    ----------
    retrieval_results : list
        List of retrieval result objects. Each result is expected to have:
        - chunk_id
        - rank
        - score
        - text
        - metadata with path, start_line, end_line
    max_context_tokens : int
        Maximum approximate token budget.
    separator : str
        Text inserted between packed chunks.

    Returns
    -------
    PackedContext
    """
    packed_chunks: List[PackedChunk] = []
    packed_text_parts: List[str] = []
    total_tokens = 0

    for result in retrieval_results:
        chunk_text = result.text.strip()
        if not chunk_text:
            continue

        header = (
            f"[Chunk rank={result.rank} score={result.score:.4f} "
            f"path={result.metadata.get('path')} "
            f"lines={result.metadata.get('start_line')}-{result.metadata.get('end_line')}]"
        )

        full_text = header + "\n" + chunk_text
        chunk_tokens = approx_token_count(full_text)

        # stop if adding this chunk exceeds the budget
        # if packed_chunks and total_tokens + chunk_tokens > max_context_tokens:
        #     break

        # # always allow the first chunk, even if large
        # if not packed_chunks and chunk_tokens > max_context_tokens:
        #     pass

        if total_tokens + chunk_tokens > max_context_tokens:
            continue

        packed_text_parts.append(full_text)
        total_tokens += chunk_tokens

        packed_chunks.append(
            PackedChunk(
                chunk_id=result.chunk_id,
                rank=result.rank,
                score=float(result.score),
                path=result.metadata.get("path", ""),
                start_line=result.metadata.get("start_line", -1),
                end_line=result.metadata.get("end_line", -1),
                token_count=chunk_tokens,
                text=chunk_text,
            )
        )

    context_text = separator.join(packed_text_parts)

    return PackedContext(
        total_tokens=total_tokens,
        max_tokens=max_context_tokens,
        num_chunks=len(packed_chunks),
        context_text=context_text,
        packed_chunks=packed_chunks,
    )


if __name__ == "__main__":
    from src.retrieval.bm25_retriever import BM25CodeRetriever

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
    results = retriever.search("function to add two numbers", top_k=3)

    packed = pack_topk_results(results, max_context_tokens=60)

    print("=" * 80)
    print("Packed summary")
    print(f"num_chunks: {packed.num_chunks}")
    print(f"total_tokens: {packed.total_tokens}")
    print("-" * 80)
    print(packed.context_text)