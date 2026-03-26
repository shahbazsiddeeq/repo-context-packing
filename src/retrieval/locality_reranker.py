from __future__ import annotations

from dataclasses import dataclass
from typing import List, Any, Dict


@dataclass
class RerankedResult:
    chunk_id: str
    original_score: float
    reranked_score: float
    rank: int
    text: str
    metadata: Dict[str, Any]


def rerank_with_locality(
    retrieval_results: List[Any],
    current_path: str,
    current_line: int,
    same_file_bonus: float = 100.0,
    line_distance_weight: float = 0.2,
) -> List[RerankedResult]:
    """
    Rerank BM25 results using locality priors:
    - strong bonus for chunks from the same file
    - smaller penalty for line distance from current cursor/target line

    This is realistic for repository completion because the current file path
    and current cursor region are known to the assistant.
    """
    scored = []

    for result in retrieval_results:
        path = result.metadata.get("path", "")
        start_line = result.metadata.get("start_line", -1)
        end_line = result.metadata.get("end_line", -1)

        score = float(result.score)

        # same-file bonus
        if path == current_path:
            score += same_file_bonus

            # line-distance bonus within same file
            if start_line != -1 and end_line != -1 and current_line > 0:
                if start_line <= current_line <= end_line:
                    distance = 0
                else:
                    distance = min(abs(current_line - start_line), abs(current_line - end_line))
                score -= line_distance_weight * distance

        scored.append(
            {
                "chunk_id": result.chunk_id,
                "original_score": float(result.score),
                "reranked_score": score,
                "text": result.text,
                "metadata": dict(result.metadata),
            }
        )

    scored.sort(key=lambda x: x["reranked_score"], reverse=True)

    reranked = []
    for i, item in enumerate(scored, start=1):
        reranked.append(
            RerankedResult(
                chunk_id=item["chunk_id"],
                original_score=item["original_score"],
                reranked_score=item["reranked_score"],
                rank=i,
                text=item["text"],
                metadata=item["metadata"],
            )
        )

    return reranked


if __name__ == "__main__":
    class DummyResult:
        def __init__(self, chunk_id, score, text, metadata):
            self.chunk_id = chunk_id
            self.score = score
            self.text = text
            self.metadata = metadata

    results = [
        DummyResult(
            "a",
            50.0,
            "chunk a",
            {"path": "file1.py", "start_line": 10, "end_line": 20},
        ),
        DummyResult(
            "b",
            55.0,
            "chunk b",
            {"path": "other.py", "start_line": 1, "end_line": 10},
        ),
        DummyResult(
            "c",
            45.0,
            "chunk c",
            {"path": "file1.py", "start_line": 100, "end_line": 110},
        ),
    ]

    reranked = rerank_with_locality(
        results,
        current_path="file1.py",
        current_line=15,
    )

    for r in reranked:
        print(
            f"rank={r.rank} chunk={r.chunk_id} "
            f"orig={r.original_score:.2f} rerank={r.reranked_score:.2f} "
            f"path={r.metadata.get('path')}"
        )