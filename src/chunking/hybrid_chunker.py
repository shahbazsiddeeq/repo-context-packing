from __future__ import annotations

from typing import List, Dict, Any, Tuple

from src.chunking.fixed_chunker import chunk_record_fixed
from src.chunking.structure_chunker import chunk_record_structure_aware
import textwrap

def _dedup_key(chunk: Dict[str, Any]) -> Tuple:
    """
    Deduplicate exact same chunk content and span.
    """
    return (
        chunk.get("path", ""),
        chunk.get("start_line", -1),
        chunk.get("end_line", -1),
        chunk.get("text", ""),
    )


def chunk_record_hybrid(
    record: Dict[str, Any],
    code_field: str = "code",
    repo_field: str = "repo",
    path_field: str = "path",
    language_field: str = "language",
    fixed_chunk_size_tokens: int = 120,
    fixed_chunk_overlap_tokens: int = 20,
    structure_max_chunk_tokens: int = 140,
    structure_overlap_tokens: int = 20,
) -> List[Dict[str, Any]]:
    """
    Hybrid chunking:
    - fixed-size chunks for local contiguous context
    - structure-aware chunks for semantic/code-boundary context

    Returns a merged, deduplicated list of chunks.
    """
    fixed_chunks = chunk_record_fixed(
        record,
        code_field=code_field,
        repo_field=repo_field,
        path_field=path_field,
        language_field=language_field,
        chunk_size_tokens=fixed_chunk_size_tokens,
        chunk_overlap_tokens=fixed_chunk_overlap_tokens,
    )

    structure_chunks = chunk_record_structure_aware(
        record,
        code_field=code_field,
        repo_field=repo_field,
        path_field=path_field,
        language_field=language_field,
        fallback_to_fixed=True,
        fixed_chunk_size_tokens=fixed_chunk_size_tokens,
        fixed_chunk_overlap_tokens=fixed_chunk_overlap_tokens,
        max_chunk_tokens=structure_max_chunk_tokens,
        overlap_tokens=structure_overlap_tokens,
    )

    merged: List[Dict[str, Any]] = []
    seen = set()

    # Keep fixed chunks first, because they are your strongest current baseline
    for chunk in fixed_chunks:
        c = dict(chunk)
        c["source_mode"] = "fixed"
        key = _dedup_key(c)
        if key in seen:
            continue
        seen.add(key)
        merged.append(c)

    # Add structure-aware chunks that are not exact duplicates
    for chunk in structure_chunks:
        c = dict(chunk)
        c["source_mode"] = "structure"
        key = _dedup_key(c)
        if key in seen:
            continue
        seen.add(key)
        merged.append(c)

    return merged


if __name__ == "__main__":
    sample_record = {
        "repo": "demo_repo",
        "path": "math_utils.py",
        "language": "python",
        "code": textwrap.dedent("""
            import os

            def add(a, b):
                return a + b

            def subtract(a, b):
                return a - b

            class MathOps:
                def multiply(self, a, b):
                    return a * b

            def very_long_function():
                x = 1
                y = 2
                z = 3
                a = 4
                b = 5
                c = 6
                d = 7
                e = 8
                f = 9
                g = 10
                h = 11
                i = 12
                return x + y + z + a + b + c + d + e + f + g + h + i
        """).strip()
    }

    chunks = chunk_record_hybrid(
        sample_record,
        fixed_chunk_size_tokens=40,
        fixed_chunk_overlap_tokens=10,
        structure_max_chunk_tokens=60,
        structure_overlap_tokens=15,
    )

    print(f"Total hybrid chunks: {len(chunks)}")
    for c in chunks:
        print("=" * 80)
        print(f"chunk_id     : {c.get('chunk_id')}")
        print(f"source_mode  : {c.get('source_mode')}")
        print(f"chunk_type   : {c.get('chunk_type', 'fixed')}")
        print(f"symbol_name  : {c.get('symbol_name', '')}")
        print(f"path         : {c.get('path')}")
        print(f"lines        : {c.get('start_line')}-{c.get('end_line')}")
        print(c.get("text", "")[:300])