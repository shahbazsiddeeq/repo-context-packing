from __future__ import annotations

from pathlib import Path
from typing import Optional
import json

from src.data.load_repoeval import load_repoeval_file

REPOEVAL_FILE = Path(
    "data/raw/repoeval/benchmark_files/line_level_completion_1k_context_codegen.test.jsonl"
)

REPO_ROOT = Path("data/raw/repoeval/repositories/line_and_api_level")


def find_repo_dir(repo_name: str) -> Optional[Path]:
    direct = REPO_ROOT / repo_name
    if direct.exists() and direct.is_dir():
        return direct

    matches = [p for p in REPO_ROOT.rglob(repo_name) if p.is_dir()]
    if matches:
        return matches[0]

    return None


def main():
    examples = load_repoeval_file(REPOEVAL_FILE, max_rows=5)

    for ex in examples:
        repo_name = ex["repo"]
        target_path = ex["path"]
        target_line = ex["line_no"]
        ground_truth = ex["ground_truth"]

        print("=" * 100)
        print(f"task_id    : {ex['task_id']}")
        print(f"repo       : {repo_name}")
        print(f"path       : {target_path}")
        print(f"line_no    : {target_line}")
        print(f"groundtruth: {ground_truth}")

        repo_dir = find_repo_dir(repo_name)
        print(f"repo_dir   : {repo_dir}")

        if repo_dir is None:
            print("[ERROR] repo directory not found")
            continue

        file_path = repo_dir / target_path
        print(f"file exists: {file_path.exists()}")

        if not file_path.exists():
            # search by filename only as fallback
            filename = Path(target_path).name
            matches = list(repo_dir.rglob(filename))
            print(f"filename-only matches: {len(matches)}")
            for m in matches[:10]:
                print("  ", m)
            continue

        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = file_path.read_text(encoding="latin-1")

        lines = text.splitlines()
        print(f"num_lines  : {len(lines)}")

        if 1 <= target_line <= len(lines):
            print("target line content:")
            print(lines[target_line - 1])
        else:
            print("[WARN] target line outside file length")

        gt = (ground_truth or "").strip()
        if gt:
            found = gt in text
            print(f"ground truth text found in file: {found}")

            if not found:
                # try stripped search line by line
                stripped_matches = []
                for i, line in enumerate(lines, start=1):
                    if gt.strip() == line.strip():
                        stripped_matches.append(i)
                print(f"exact stripped line matches: {stripped_matches[:10]}")
            else:
                # print first matching line number
                for i, line in enumerate(lines, start=1):
                    if gt in line:
                        print(f"ground truth first appears at line: {i}")
                        break

        print("-" * 100)
        start = max(1, target_line - 3)
        end = min(len(lines), target_line + 3)
        print(f"context lines {start}-{end}:")
        for i in range(start, end + 1):
            print(f"{i:>5}: {lines[i - 1]}")


if __name__ == "__main__":
    main()