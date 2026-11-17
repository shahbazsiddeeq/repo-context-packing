"""Recover the text of the chunk placed in each prompt of the five-condition run.

Rebuilds the per-instance retrieval index exactly as
scripts/run_repoeval_completion_model.py does (target file cut at the completion
line, fixed 120/20 chunks, BM25 top-10, locality reranking, nearest window,
cross-file chunk from the top 50), checks the BM25 and locality spans against the
logged per-instance CSV, and writes one JSON line per instance with the prefix and
the four chunk texts. CPU only.

Usage: python scripts/extract_prompt_chunks.py <per_example.csv> <out.jsonl> [workers]
"""
from __future__ import annotations

import csv
import json
import sys
from multiprocessing import Pool

from src.data.load_repoeval import load_repoeval_file
from src.retrieval.bm25_retriever import BM25CodeRetriever
from src.retrieval.locality_reranker import rerank_with_locality
from scripts.run_repoeval_first_experiment import (
    REPOEVAL_FILE,
    find_repo_dir,
    load_python_files_from_repo,
)
from scripts.run_repoeval_completion_model import (
    CROSSFILE_SEARCH_K,
    LINE_DISTANCE_WEIGHT,
    MAX_FILES_PER_REPO,
    SAME_FILE_BONUS,
    TOP_K,
    build_fixed_chunks,
)


def _path(r):
    return r.metadata.get("path", "")


def process(ex):
    repo_dir = find_repo_dir(ex["repo"])
    records = load_python_files_from_repo(
        repo_dir,
        max_files=MAX_FILES_PER_REPO,
        required_path=ex["path"],
        truncate_line=ex["line_no"],
    )
    chunks = build_fixed_chunks(records)
    retriever = BM25CodeRetriever(chunks)
    bm25 = retriever.search(ex["query"], top_k=TOP_K)
    loc = rerank_with_locality(
        bm25,
        current_path=ex["path"],
        current_line=ex["line_no"] + 1,
        same_file_bonus=SAME_FILE_BONUS,
        line_distance_weight=LINE_DISTANCE_WEIGHT,
    )
    same_file = [c for c in chunks if c.get("path") == ex["path"]]
    nearest = max(same_file, key=lambda c: c.get("end_line", -1)) if same_file else None
    wide = retriever.search(ex["query"], top_k=CROSSFILE_SEARCH_K)
    cross = next((r for r in wide if _path(r) != ex["path"]), None)
    return {
        "task_id": ex["task_id"],
        "prefix": ex.get("prompt", ex["query"]),
        "bm25": {"path": _path(bm25[0]), "start": bm25[0].metadata["start_line"],
                 "end": bm25[0].metadata["end_line"], "text": bm25[0].text},
        "locality": {"path": _path(loc[0]), "start": loc[0].metadata["start_line"],
                     "end": loc[0].metadata["end_line"], "text": loc[0].text},
        "nearest": None if nearest is None else {
            "path": ex["path"], "start": nearest["start_line"],
            "end": nearest["end_line"], "text": nearest["text"]},
        "crossfile": None if cross is None else {
            "path": _path(cross), "start": cross.metadata["start_line"],
            "end": cross.metadata["end_line"], "text": cross.text},
    }


def main():
    csv_path, out_path = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    with open(csv_path, encoding="utf-8") as f:
        logged = {r["task_id"]: r for r in csv.DictReader(f)}
    examples = [e for e in load_repoeval_file(REPOEVAL_FILE, max_rows=1600)
                if e["task_id"] in logged]
    mismatches = 0
    with Pool(workers) as pool, open(out_path, "w", encoding="utf-8") as out:
        for i, rec in enumerate(pool.imap(process, examples, chunksize=4), 1):
            row = logged[rec["task_id"]]
            for arm in ("bm25", "locality"):
                c = rec[arm]
                if (c["path"], str(c["start"]), str(c["end"])) != (
                        row[f"{arm}_top1_path"], row[f"{arm}_top1_start"], row[f"{arm}_top1_end"]):
                    mismatches += 1
            if rec["crossfile"] is not None and rec["crossfile"]["path"] != row["crossfile_path"]:
                mismatches += 1
            out.write(json.dumps(rec) + "\n")
            if i % 100 == 0:
                print(f"{i}/{len(examples)} done, span mismatches so far: {mismatches}", flush=True)
    print(f"Finished {len(examples)} instances; span mismatches with the logged run: {mismatches}")


if __name__ == "__main__":
    main()
