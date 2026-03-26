from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_jsonl(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(json.loads(line))
    return rows


def build_code_query_from_prompt(
    prompt: str,
    max_lines: int = 20,
    max_chars: int = 1500,
) -> str:
    """
    Build a retrieval query from the trailing part of the prompt.

    For RepoEval line-level tasks, the prompt is code context rather than
    natural language. We use the last few non-empty lines as the retrieval
    query signal.
    """
    if not prompt.strip():
        return ""

    lines = [line.rstrip() for line in prompt.splitlines()]
    non_empty = [line for line in lines if line.strip()]

    tail_lines = non_empty[-max_lines:]
    query = "\n".join(tail_lines)

    if len(query) > max_chars:
        query = query[-max_chars:]

    return query


def normalize_repoeval_record(row: Dict[str, Any]) -> Dict[str, Any]:
    metadata = row.get("metadata", {})
    fpath_tuple = metadata.get("fpath_tuple", [])

    repo = fpath_tuple[0] if len(fpath_tuple) > 0 else "unknown_repo"
    rel_path = "/".join(fpath_tuple[1:]) if len(fpath_tuple) > 1 else "unknown_path"

    prompt = row.get("prompt", "")
    ground_truth = metadata.get("ground_truth", "")

    return {
        "task_id": metadata.get("task_id", ""),
        "repo": repo,
        "path": rel_path,
        "language": "python",
        "prompt": prompt,
        "query": build_code_query_from_prompt(prompt),
        "ground_truth": ground_truth,
        "line_no": metadata.get("line_no", -1),
        "context_start_lineno": metadata.get("context_start_lineno", -1),
        "fpath_tuple": fpath_tuple,
    }


def load_repoeval_file(
    path: str | Path,
    max_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    path = Path(path)
    rows = load_jsonl(path, max_rows=max_rows)
    return [normalize_repoeval_record(r) for r in rows]


if __name__ == "__main__":
    default_path = Path(
        "data/raw/repoeval/benchmark_files/line_level_completion_1k_context_codegen.test.jsonl"
    )

    if not default_path.exists():
        raise FileNotFoundError(f"Missing RepoEval file: {default_path}")

    rows = load_repoeval_file(default_path, max_rows=3)

    print(f"Loaded rows: {len(rows)}")
    for i, row in enumerate(rows, start=1):
        print("=" * 100)
        print(f"Example {i}")
        print(f"task_id: {row['task_id']}")
        print(f"repo: {row['repo']}")
        print(f"path: {row['path']}")
        print(f"line_no: {row['line_no']}")
        print("-" * 80)
        print("Query tail:")
        print(row["query"][:1200])
        print("-" * 80)
        print("Ground truth:")
        print(row["ground_truth"])