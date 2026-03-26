from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Set


def approx_token_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


@dataclass
class NeighborPackedChunk:
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
class NeighborPackedContext:
    total_tokens: int
    max_tokens: int
    num_chunks: int
    context_text: str
    packed_chunks: List[NeighborPackedChunk]

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


def pack_neighbor_aware_results(
    retrieval_results: List[Any],
    all_chunks: List[Dict[str, Any]],
    max_context_tokens: int = 200,
    max_neighbors_per_side: int = 1,
    separator: str = "\n\n" + "=" * 80 + "\n\n",
) -> NeighborPackedContext:
    by_path = build_same_file_index(all_chunks)

    packed_chunks: List[NeighborPackedChunk] = []
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

        # Add retrieved chunk
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
                NeighborPackedChunk(
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

        # Add same-file neighbors
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
                NeighborPackedChunk(
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

    return NeighborPackedContext(
        total_tokens=total_tokens,
        max_tokens=max_context_tokens,
        num_chunks=len(packed_chunks),
        context_text=context_text,
        packed_chunks=packed_chunks,
    )