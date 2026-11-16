from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from src.data.load_repoeval import load_repoeval_file
from src.chunking.fixed_chunker import chunk_record_fixed
from src.chunking.structure_chunker import chunk_record_structure_aware
from src.chunking.hybrid_chunker import chunk_record_hybrid
from src.retrieval.bm25_retriever import BM25CodeRetriever


REPOEVAL_FILE = Path(
    "data/raw/repoeval/benchmark_files/line_level_completion_1k_context_codegen.test.jsonl"
)

REPO_ROOT = Path("data/raw/repoeval/repositories/line_and_api_level")


def reciprocal_rank(results, gold_chunk_id: str) -> float:
    for r in results:
        if r.chunk_id == gold_chunk_id:
            return 1.0 / r.rank
    return 0.0


def recall_at_k(results, gold_chunk_id: str, k: int) -> float:
    return 1.0 if any(r.chunk_id == gold_chunk_id for r in results[:k]) else 0.0


def find_repo_dir(repo_name: str) -> Optional[Path]:
    """
    Try a few common layouts.
    """
    direct = REPO_ROOT / repo_name
    if direct.exists() and direct.is_dir():
        return direct

    # fallback: search recursively
    matches = [p for p in REPO_ROOT.rglob(repo_name) if p.is_dir()]
    if matches:
        return matches[0]

    return None


def load_python_files_from_repo(
    repo_dir: Path,
    max_files: int = 200,
    required_path: str | None = None,
    truncate_line: int | None = None,
) -> List[Dict[str, Any]]:
    # truncate_line: 0-based index of the completion (reference) line in the
    # target file. When set, the target file (required_path) is cut to the lines
    # strictly before that line before it enters the retrieval pool. This removes
    # the reference line and everything after it from same-file chunks, which
    # otherwise leaks the answer into the retrieved context.
    py_files = sorted(repo_dir.rglob("*.py"))

    # Always include the target file first if provided
    selected_files = []
    required_abs = None

    if required_path:
        candidate = repo_dir / required_path
        if candidate.exists():
            required_abs = candidate.resolve()
            selected_files.append(candidate)

    for fpath in py_files:
        if required_abs is not None and fpath.resolve() == required_abs:
            continue
        selected_files.append(fpath)
        if len(selected_files) >= max_files:
            break

    records = []

    for fpath in selected_files:
        try:
            code = fpath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                code = fpath.read_text(encoding="latin-1")
            except Exception:
                continue
        except Exception:
            continue

        rel_path = str(fpath.relative_to(repo_dir))

        if (
            truncate_line is not None
            and required_abs is not None
            and fpath.resolve() == required_abs
        ):
            code = "\n".join(code.splitlines()[:truncate_line])

        records.append(
            {
                "repo": repo_dir.name,
                "path": rel_path,
                "language": "python",
                "code": code,
            }
        )

    return records


# def build_repo_chunks(
#     repo_records: List[Dict[str, Any]],
#     mode: str = "fixed",
# ) -> List[Dict[str, Any]]:
#     all_chunks: List[Dict[str, Any]] = []

#     for record in repo_records:
#         if not record["code"].strip():
#             continue

#         if mode == "fixed":
#             chunks = chunk_record_fixed(
#                 record,
#                 code_field="code",
#                 repo_field="repo",
#                 path_field="path",
#                 language_field="language",
#                 chunk_size_tokens=120,
#                 chunk_overlap_tokens=20,
#             )
#         elif mode == "structure":
#             chunks = chunk_record_structure_aware(
#                 record,
#                 code_field="code",
#                 repo_field="repo",
#                 path_field="path",
#                 language_field="language",
#                 fallback_to_fixed=True,
#                 fixed_chunk_size_tokens=120,
#                 fixed_chunk_overlap_tokens=20,
#                 max_chunk_tokens=140,
#                 overlap_tokens=20,
#             )
#         else:
#             raise ValueError(f"Unknown mode: {mode}")

#         all_chunks.extend(chunks)

#     return all_chunks

def build_repo_chunks(
    repo_records: List[Dict[str, Any]],
    mode: str = "fixed",
) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []

    for record in repo_records:
        if not record["code"].strip():
            continue

        if mode == "fixed":
            chunks = chunk_record_fixed(
                record,
                code_field="code",
                repo_field="repo",
                path_field="path",
                language_field="language",
                chunk_size_tokens=120,
                chunk_overlap_tokens=20,
            )

        elif mode == "structure":
            chunks = chunk_record_structure_aware(
                record,
                code_field="code",
                repo_field="repo",
                path_field="path",
                language_field="language",
                fallback_to_fixed=True,
                fixed_chunk_size_tokens=120,
                fixed_chunk_overlap_tokens=20,
                max_chunk_tokens=140,
                overlap_tokens=20,
            )

        elif mode == "hybrid":
            chunks = chunk_record_hybrid(
                record,
                code_field="code",
                repo_field="repo",
                path_field="path",
                language_field="language",
                fixed_chunk_size_tokens=120,
                fixed_chunk_overlap_tokens=20,
                structure_max_chunk_tokens=140,
                structure_overlap_tokens=20,
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        all_chunks.extend(chunks)

    return all_chunks


def find_gold_chunk_id(
    chunks: List[Dict[str, Any]],
    target_path: str,
    target_line: int,
    ground_truth: str,
) -> Optional[str]:
    gt = (ground_truth or "").strip()

    # 1) strongest match: same file + exact line content after stripping indentation
    if gt:
        for chunk in chunks:
            if chunk.get("path") != target_path:
                continue

            chunk_lines = [line.strip() for line in chunk.get("text", "").splitlines()]
            if gt in chunk_lines:
                return chunk["chunk_id"]

    # 2) RepoEval line-level usually points to the line BEFORE the missing line
    for candidate_line in [target_line + 1, target_line, target_line - 1]:
        for chunk in chunks:
            if chunk.get("path") != target_path:
                continue
            start_line = chunk.get("start_line", -1)
            end_line = chunk.get("end_line", -1)
            if start_line <= candidate_line <= end_line:
                return chunk["chunk_id"]

    return None


def evaluate_repoeval_subset(
    mode: str = "fixed",
    max_examples: int = 5,
    top_k: int = 10,
) -> Dict[str, Any]:
    records = load_repoeval_file(REPOEVAL_FILE, max_rows=max_examples)

    rr_scores = []
    r1_scores = []
    r5_scores = []
    usable = 0

    for ex in records:
        repo_name = ex["repo"]
        target_path = ex["path"]
        target_line = ex["line_no"]
        query = ex["query"]

        repo_dir = find_repo_dir(repo_name)
        if repo_dir is None:
            print(f"[skip] repo not found: {repo_name}")
            continue

        # repo_records = load_python_files_from_repo(repo_dir, max_files=200)
        repo_records = load_python_files_from_repo(
                            repo_dir,
                            max_files=200,
                            required_path=target_path,
                        )
        if not repo_records:
            print(f"[skip] no python files: {repo_name}")
            continue

        all_chunks = build_repo_chunks(repo_records, mode=mode)
        if not all_chunks:
            print(f"[skip] no chunks: {repo_name}")
            continue

        gold_chunk_id = find_gold_chunk_id(
                            all_chunks,
                            target_path,
                            target_line,
                            ex["ground_truth"],
                        )
        if gold_chunk_id is None:
            print(f"[skip] gold line not covered: {repo_name} | {target_path}:{target_line}")
            continue

        retriever = BM25CodeRetriever(all_chunks)
        results = retriever.search(query, top_k=top_k)

        rr = reciprocal_rank(results, gold_chunk_id)
        r1 = recall_at_k(results, gold_chunk_id, 1)
        r5 = recall_at_k(results, gold_chunk_id, 5)

        rr_scores.append(rr)
        r1_scores.append(r1)
        r5_scores.append(r5)
        usable += 1

        print("=" * 100)
        print(f"task_id   : {ex['task_id']}")
        print(f"repo      : {repo_name}")
        print(f"path      : {target_path}")
        print(f"line_no   : {target_line}")
        print(f"mode      : {mode}")
        print(f"gold_chunk: {gold_chunk_id}")
        print(f"RR        : {rr:.4f}")
        print(f"R@1       : {r1:.1f}")
        print(f"R@5       : {r5:.1f}")

        for r in results[:5]:
            marker = " <-- GOLD" if r.chunk_id == gold_chunk_id else ""
            print(
                f"rank={r.rank} score={r.score:.4f} "
                f"path={r.metadata.get('path')} "
                f"lines={r.metadata.get('start_line')}-{r.metadata.get('end_line')}"
                f"{marker}"
            )

    if usable == 0:
        raise RuntimeError("No usable RepoEval examples were evaluated.")

    return {
        "mode": mode,
        "num_examples": max_examples,
        "usable_examples": usable,
        "MRR": sum(rr_scores) / usable,
        "Recall@1": sum(r1_scores) / usable,
        "Recall@5": sum(r5_scores) / usable,
    }


def print_summary(metrics: Dict[str, Any]) -> None:
    print("\n" + "#" * 100)
    print("RepoEval summary")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:16s}: {v:.4f}")
        else:
            print(f"{k:16s}: {v}")


def main():
    # metrics = evaluate_repoeval_subset(
    #     mode="fixed",
    #     max_examples=5,
    #     top_k=10,
    # )
    metrics = evaluate_repoeval_subset(
        mode="structure",
        max_examples=20,
        top_k=10,
    )
    print_summary(metrics)


if __name__ == "__main__":
    main()