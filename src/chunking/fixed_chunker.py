from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import math


@dataclass
class CodeChunk:
    chunk_id: str
    repo: str
    path: str
    language: str
    start_line: int
    end_line: int
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _line_token_count(line: str) -> int:
    """
    Very simple token approximation for baseline chunking.
    Later, if needed, we can replace this with a real tokenizer.
    """
    return len(line.strip().split())


def chunk_code_fixed(
    code: str,
    repo: str,
    path: str,
    language: str = "python",
    chunk_size_tokens: int = 300,
    chunk_overlap_tokens: int = 50,
) -> List[CodeChunk]:
    """
    Split code into fixed-size overlapping chunks using an approximate
    whitespace-token budget.

    Parameters
    ----------
    code : str
        Full source code text.
    repo : str
        Repository name or identifier.
    path : str
        File path inside repository.
    language : str
        Programming language label.
    chunk_size_tokens : int
        Approximate max tokens per chunk.
    chunk_overlap_tokens : int
        Approximate token overlap between consecutive chunks.

    Returns
    -------
    List[CodeChunk]
    """
    if not code.strip():
        return []

    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    lines = code.splitlines()
    chunks: List[CodeChunk] = []

    start_idx = 0
    chunk_num = 0

    while start_idx < len(lines):
        token_count = 0
        end_idx = start_idx
        current_lines = []

        while end_idx < len(lines):
            line = lines[end_idx]
            next_count = _line_token_count(line)

            # Always include at least one line
            if current_lines and token_count + next_count > chunk_size_tokens:
                break

            current_lines.append(line)
            token_count += next_count
            end_idx += 1

        # chunk_text = "\n".join(current_lines).strip()
        chunk_text = "\n".join(current_lines).rstrip()
        if chunk_text:
            chunk_id = f"{repo}:{path}:chunk{chunk_num}"
            chunks.append(
                CodeChunk(
                    chunk_id=chunk_id,
                    repo=repo,
                    path=path,
                    language=language,
                    start_line=start_idx + 1,
                    end_line=end_idx,
                    text=chunk_text,
                )
            )
            chunk_num += 1

        if end_idx >= len(lines):
            break

        # Move start index backward by approximate overlap
        overlap_tokens = 0
        new_start = end_idx - 1
        while new_start > start_idx:
            overlap_tokens += _line_token_count(lines[new_start])
            if overlap_tokens >= chunk_overlap_tokens:
                break
            new_start -= 1

        # Prevent infinite loop
        if new_start <= start_idx:
            new_start = end_idx

        start_idx = new_start

    return chunks


def chunk_record_fixed(
    record: Dict[str, Any],
    code_field: str = "code",
    repo_field: str = "repo",
    path_field: str = "path",
    language_field: str = "language",
    chunk_size_tokens: int = 300,
    chunk_overlap_tokens: int = 50,
) -> List[Dict[str, Any]]:
    """
    Chunk one dataset record and return list of dictionaries.
    """
    code = record.get(code_field, "")
    repo = record.get(repo_field, "unknown_repo")
    path = record.get(path_field, "unknown_path")
    language = record.get(language_field, "python")

    chunks = chunk_code_fixed(
        code=code,
        repo=repo,
        path=path,
        language=language,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )
    return [c.to_dict() for c in chunks]


if __name__ == "__main__":
    sample_code = """
import os

def add(a, b):
    return a + b

def sub(a, b):
    return a - b

class MathOps:
    def mul(self, a, b):
        return a * b
""".strip()

    chunks = chunk_code_fixed(
        code=sample_code,
        repo="demo_repo",
        path="math_utils.py",
        language="python",
        chunk_size_tokens=20,
        chunk_overlap_tokens=5,
    )

    for c in chunks:
        print("=" * 60)
        print(c.chunk_id)
        print(f"Lines: {c.start_line}-{c.end_line}")
        print(c.text)