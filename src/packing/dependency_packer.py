from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Set
import re


def approx_token_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def extract_identifiers(text: str) -> Set[str]:
    """
    Very simple identifier extraction.
    """
    if not text:
        return set()
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
    return set(tokens)


@dataclass
class DependencyPackedChunk:
    chunk_id: str
    rank: int
    score: float
    path: str
    start_line: int
    end_line: int
    token_count: int
    text: str
    source: str  # "retrieved" or "dependency"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DependencyPackedContext:
    total_tokens: int
    max_tokens: int
    num_chunks: int
    context_text: str
    packed_chunks: List[DependencyPackedChunk]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "num_chunks": self.num_chunks,
            "context_text": self.context_text,
            "packed_chunks": [c.to_dict() for c in self.packed_chunks],
        }


def build_symbol_index(all_chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Map symbol_name -> chunks that define that symbol.
    Only works for structure-aware chunks or any chunk carrying symbol_name metadata.
    """
    index: Dict[str, List[Dict[str, Any]]] = {}

    for chunk in all_chunks:
        symbol_name = chunk.get("symbol_name", "")
        if not symbol_name:
            continue
        index.setdefault(symbol_name, []).append(chunk)

    return index


def format_chunk_header(rank: int, score: float, path: str, start_line: int, end_line: int, source: str) -> str:
    return (
        f"[Chunk rank={rank} score={score:.4f} source={source} "
        f"path={path} lines={start_line}-{end_line}]"
    )


def pack_dependency_aware_results(
    retrieval_results: List[Any],
    all_chunks: List[Dict[str, Any]],
    max_context_tokens: int = 400,
    max_dependency_expansions_per_chunk: int = 2,
    separator: str = "\n\n" + "=" * 80 + "\n\n",
) -> DependencyPackedContext:
    """
    Pack top retrieved chunks and add lightweight dependency chunks
    whose symbol_name appears in the identifiers of retrieved text.

    Parameters
    ----------
    retrieval_results : list
        RetrievalResult-like objects from the retriever.
    all_chunks : list
        Full chunk collection, ideally structure-aware chunks with symbol_name.
    max_context_tokens : int
        Approximate token budget.
    max_dependency_expansions_per_chunk : int
        Maximum number of dependency chunks added per retrieved chunk.
    """
    symbol_index = build_symbol_index(all_chunks)

    packed_chunks: List[DependencyPackedChunk] = []
    packed_text_parts: List[str] = []
    total_tokens = 0
    seen_chunk_ids: Set[str] = set()

    for result in retrieval_results:
        # 1. Add the retrieved chunk itself
        base_text = result.text.strip()
        if not base_text:
            continue

        if result.chunk_id not in seen_chunk_ids:
            header = format_chunk_header(
                rank=result.rank,
                score=float(result.score),
                path=result.metadata.get("path", ""),
                start_line=result.metadata.get("start_line", -1),
                end_line=result.metadata.get("end_line", -1),
                source="retrieved",
            )
            full_text = header + "\n" + base_text
            chunk_tokens = approx_token_count(full_text)

            # if packed_chunks and total_tokens + chunk_tokens > max_context_tokens:
            #     break

            # if not packed_chunks and chunk_tokens > max_context_tokens:
            #     pass

            if total_tokens + chunk_tokens > max_context_tokens:
                continue

            packed_text_parts.append(full_text)
            total_tokens += chunk_tokens
            seen_chunk_ids.add(result.chunk_id)

            packed_chunks.append(
                DependencyPackedChunk(
                    chunk_id=result.chunk_id,
                    rank=result.rank,
                    score=float(result.score),
                    path=result.metadata.get("path", ""),
                    start_line=result.metadata.get("start_line", -1),
                    end_line=result.metadata.get("end_line", -1),
                    token_count=chunk_tokens,
                    text=base_text,
                    source="retrieved",
                )
            )

        # 2. Add dependency chunks based on identifiers found in retrieved text
        identifiers = extract_identifiers(base_text)
        dependency_added = 0

        for ident in sorted(identifiers):
            if dependency_added >= max_dependency_expansions_per_chunk:
                break

            candidates = symbol_index.get(ident, [])
            for dep_chunk in candidates:
                dep_chunk_id = dep_chunk.get("chunk_id")
                if not dep_chunk_id or dep_chunk_id in seen_chunk_ids:
                    continue

                dep_text = dep_chunk.get("text", "").strip()
                if not dep_text:
                    continue

                dep_header = format_chunk_header(
                    rank=result.rank,
                    score=float(result.score),
                    path=dep_chunk.get("path", ""),
                    start_line=dep_chunk.get("start_line", -1),
                    end_line=dep_chunk.get("end_line", -1),
                    source="dependency",
                )
                dep_full_text = dep_header + "\n" + dep_text
                dep_tokens = approx_token_count(dep_full_text)

                if total_tokens + dep_tokens > max_context_tokens:
                    continue

                packed_text_parts.append(dep_full_text)
                total_tokens += dep_tokens
                seen_chunk_ids.add(dep_chunk_id)
                dependency_added += 1

                packed_chunks.append(
                    DependencyPackedChunk(
                        chunk_id=dep_chunk_id,
                        rank=result.rank,
                        score=float(result.score),
                        path=dep_chunk.get("path", ""),
                        start_line=dep_chunk.get("start_line", -1),
                        end_line=dep_chunk.get("end_line", -1),
                        token_count=dep_tokens,
                        text=dep_text,
                        source="dependency",
                    )
                )

                break

    context_text = separator.join(packed_text_parts)

    return DependencyPackedContext(
        total_tokens=total_tokens,
        max_tokens=max_context_tokens,
        num_chunks=len(packed_chunks),
        context_text=context_text,
        packed_chunks=packed_chunks,
    )


if __name__ == "__main__":
    from src.retrieval.bm25_retriever import BM25CodeRetriever

    all_chunks = [
        {
            "chunk_id": "repo1:file1.py:struct0",
            "repo": "repo1",
            "path": "file1.py",
            "language": "python",
            "chunk_type": "function",
            "symbol_name": "helper",
            "start_line": 1,
            "end_line": 2,
            "text": "def helper(x):\n    return x + 1",
        },
        {
            "chunk_id": "repo1:file1.py:struct1",
            "repo": "repo1",
            "path": "file1.py",
            "language": "python",
            "chunk_type": "function",
            "symbol_name": "main_func",
            "start_line": 4,
            "end_line": 6,
            "text": "def main_func(y):\n    z = helper(y)\n    return z",
        },
        {
            "chunk_id": "repo1:file2.py:struct0",
            "repo": "repo1",
            "path": "file2.py",
            "language": "python",
            "chunk_type": "function",
            "symbol_name": "other_func",
            "start_line": 1,
            "end_line": 2,
            "text": "def other_func(a):\n    return a * 2",
        },
    ]

    retriever = BM25CodeRetriever(all_chunks)
    results = retriever.search("function that calls helper", top_k=2)

    packed = pack_dependency_aware_results(
        retrieval_results=results,
        all_chunks=all_chunks,
        max_context_tokens=100,
        max_dependency_expansions_per_chunk=2,
    )

    print("=" * 80)
    print("Dependency-aware packed summary")
    print(f"num_chunks: {packed.num_chunks}")
    print(f"total_tokens: {packed.total_tokens}")
    print("-" * 80)
    for c in packed.packed_chunks:
        print(f"{c.source:10s} | {c.chunk_id}")
    print("=" * 80)
    print(packed.context_text)