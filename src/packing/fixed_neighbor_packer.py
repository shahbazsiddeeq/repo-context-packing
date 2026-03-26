from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Set


def approx_token_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


@dataclass
class FixedNeighborPackedChunk:
    chunk_id: str
    rank: int
    score: float
    path: str
    start_line: int
    end_line: int
    token_count: int
    text: str
    source: str  # "retrieved" or "neighbor"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FixedNeighborPackedContext:
    total_tokens: int
    max_tokens: int
    num_chunks: int
    context_text: str
    packed_chunks: List[FixedNeighborPackedChunk]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "num_chunks": self.num_chunks,
            "context_text": self.context_text,
            "packed_chunks": [c.to_dict() for c in self.packed_chunks],
        }


def format_chunk_header(rank: int, score: float, path: str, start_line: int, end_line: int, source: str) -> str:
    return (
        f"[Chunk rank={rank} score={score:.4f} source={source} "
        f"path={path} lines={start_line}-{end_line}]"
    )


def build_same_file_index(all_chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_path: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in all_chunks:
        path = chunk.get("path", "")
        if not path:
            continue
        by_path.setdefault(path, []).append(chunk)

    for path in by_path:
        by_path[path] = sorted(
            by_path[path],
            key=lambda c: (c.get("start_line", -1), c.get("end_line", -1))
        )
    return by_path


def pack_fixed_neighbor_results(
    retrieval_results: List[Any],
    all_chunks: List[Dict[str, Any]],
    max_context_tokens: int = 200,
    max_neighbors_per_side: int = 1,
    separator: str = "\n\n" + "=" * 80 + "\n\n",
) -> FixedNeighborPackedContext:
    """
    Pack retrieved fixed chunks and expand with previous/next chunks
    from the same file under a hard token budget.
    """
    by_path = build_same_file_index(all_chunks)

    packed_chunks: List[FixedNeighborPackedChunk] = []
    packed_text_parts: List[str] = []
    total_tokens = 0
    seen_chunk_ids: Set[str] = set()

    for result in retrieval_results:
        base_text = result.text.strip()
        if not base_text:
            continue

        base_path = result.metadata.get("path", "")
        base_start = result.metadata.get("start_line", -1)
        base_end = result.metadata.get("end_line", -1)

        # add retrieved chunk first
        if result.chunk_id not in seen_chunk_ids:
            header = format_chunk_header(
                rank=result.rank,
                score=float(result.score),
                path=base_path,
                start_line=base_start,
                end_line=base_end,
                source="retrieved",
            )
            full_text = header + "\n" + base_text
            chunk_tokens = approx_token_count(full_text)

            if total_tokens + chunk_tokens > max_context_tokens:
                continue

            packed_text_parts.append(full_text)
            total_tokens += chunk_tokens
            seen_chunk_ids.add(result.chunk_id)

            packed_chunks.append(
                FixedNeighborPackedChunk(
                    chunk_id=result.chunk_id,
                    rank=result.rank,
                    score=float(result.score),
                    path=base_path,
                    start_line=base_start,
                    end_line=base_end,
                    token_count=chunk_tokens,
                    text=base_text,
                    source="retrieved",
                )
            )

        # add neighbors from same file
        same_file_chunks = by_path.get(base_path, [])
        pos = None
        for i, chunk in enumerate(same_file_chunks):
            if chunk.get("chunk_id") == result.chunk_id:
                pos = i
                break

        if pos is None:
            continue

        neighbor_positions = []
        for offset in range(1, max_neighbors_per_side + 1):
            if pos - offset >= 0:
                neighbor_positions.append(pos - offset)
            if pos + offset < len(same_file_chunks):
                neighbor_positions.append(pos + offset)

        for np in neighbor_positions:
            neighbor = same_file_chunks[np]
            neighbor_id = neighbor.get("chunk_id")
            if not neighbor_id or neighbor_id in seen_chunk_ids:
                continue

            neighbor_text = neighbor.get("text", "").strip()
            if not neighbor_text:
                continue

            header = format_chunk_header(
                rank=result.rank,
                score=float(result.score),
                path=neighbor.get("path", ""),
                start_line=neighbor.get("start_line", -1),
                end_line=neighbor.get("end_line", -1),
                source="neighbor",
            )
            full_text = header + "\n" + neighbor_text
            chunk_tokens = approx_token_count(full_text)

            if total_tokens + chunk_tokens > max_context_tokens:
                continue

            packed_text_parts.append(full_text)
            total_tokens += chunk_tokens
            seen_chunk_ids.add(neighbor_id)

            packed_chunks.append(
                FixedNeighborPackedChunk(
                    chunk_id=neighbor_id,
                    rank=result.rank,
                    score=float(result.score),
                    path=neighbor.get("path", ""),
                    start_line=neighbor.get("start_line", -1),
                    end_line=neighbor.get("end_line", -1),
                    token_count=chunk_tokens,
                    text=neighbor_text,
                    source="neighbor",
                )
            )

    context_text = separator.join(packed_text_parts)

    return FixedNeighborPackedContext(
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
            "chunk_id": "repo1:file.py:chunk0",
            "repo": "repo1",
            "path": "file.py",
            "language": "python",
            "start_line": 1,
            "end_line": 4,
            "text": "import os\n\ndef helper(x):\n    return x + 1",
        },
        {
            "chunk_id": "repo1:file.py:chunk1",
            "repo": "repo1",
            "path": "file.py",
            "language": "python",
            "start_line": 5,
            "end_line": 8,
            "text": "def main(y):\n    z = helper(y)\n    return z",
        },
        {
            "chunk_id": "repo1:file.py:chunk2",
            "repo": "repo1",
            "path": "file.py",
            "language": "python",
            "start_line": 9,
            "end_line": 12,
            "text": "print(main(3))",
        },
    ]

    retriever = BM25CodeRetriever(demo_chunks)
    results = retriever.search("function using helper", top_k=2)

    packed = pack_fixed_neighbor_results(
        retrieval_results=results,
        all_chunks=demo_chunks,
        max_context_tokens=100,
        max_neighbors_per_side=1,
    )

    print("=" * 80)
    print("Fixed-neighbor packed summary")
    print(f"num_chunks: {packed.num_chunks}")
    print(f"total_tokens: {packed.total_tokens}")
    print("-" * 80)
    for c in packed.packed_chunks:
        print(f"{c.source:10s} | {c.chunk_id}")
    print("=" * 80)
    print(packed.context_text)