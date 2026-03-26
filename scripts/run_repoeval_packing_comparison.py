# # from __future__ import annotations

# # from pathlib import Path
# # from typing import Dict, Any, List

# # from src.data.load_repoeval import load_repoeval_file
# # from src.retrieval.bm25_retriever import BM25CodeRetriever
# # from src.packing.topk_packer import pack_topk_results
# # from src.packing.dependency_packer import pack_dependency_aware_results

# # from scripts.run_repoeval_first_experiment import (
# #     REPOEVAL_FILE,
# #     find_repo_dir,
# #     load_python_files_from_repo,
# #     build_repo_chunks,
# #     find_gold_chunk_id,
# # )


# # BUDGETS = [80, 120, 200]


# # def packed_contains_gold_chunk(packed, gold_chunk_id: str) -> bool:
# #     return any(c.chunk_id == gold_chunk_id for c in packed.packed_chunks)


# # def packed_contains_gold_line(packed, ground_truth: str) -> bool:
# #     gt = (ground_truth or "").strip()
# #     if not gt:
# #         return False

# #     packed_lines = [line.strip() for line in packed.context_text.splitlines()]
# #     return gt in packed_lines


# # def init_metrics() -> Dict[str, float]:
# #     return {
# #         "usable_examples": 0,
# #         "gold_chunk_hit": 0,
# #         "gold_line_hit": 0,
# #         "avg_tokens_sum": 0.0,
# #         "avg_chunks_sum": 0.0,
# #     }


# # def finalize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
# #     n = metrics["usable_examples"]
# #     if n == 0:
# #         return {
# #             "usable_examples": 0,
# #             "GoldChunkPackedRate": 0.0,
# #             "GoldLinePackedRate": 0.0,
# #             "AvgPackedTokens": 0.0,
# #             "AvgPackedChunks": 0.0,
# #         }

# #     return {
# #         "usable_examples": int(n),
# #         "GoldChunkPackedRate": metrics["gold_chunk_hit"] / n,
# #         "GoldLinePackedRate": metrics["gold_line_hit"] / n,
# #         "AvgPackedTokens": metrics["avg_tokens_sum"] / n,
# #         "AvgPackedChunks": metrics["avg_chunks_sum"] / n,
# #     }


# # def print_summary(title: str, metrics: Dict[str, float]) -> None:
# #     print("\n" + "=" * 100)
# #     print(title)
# #     print("-" * 100)
# #     for k, v in metrics.items():
# #         if isinstance(v, float):
# #             print(f"{k:20s}: {v:.4f}")
# #         else:
# #             print(f"{k:20s}: {v}")


# # def main():
# #     examples = load_repoeval_file(REPOEVAL_FILE, max_rows=5)

# #     topk_metrics_by_budget = {b: init_metrics() for b in BUDGETS}
# #     dep_metrics_by_budget = {b: init_metrics() for b in BUDGETS}

# #     for ex in examples:
# #         repo_name = ex["repo"]
# #         target_path = ex["path"]
# #         target_line = ex["line_no"]
# #         query = ex["query"]
# #         ground_truth = ex["ground_truth"]

# #         repo_dir = find_repo_dir(repo_name)
# #         if repo_dir is None:
# #             print(f"[skip] repo not found: {repo_name}")
# #             continue

# #         repo_records = load_python_files_from_repo(
# #             repo_dir,
# #             max_files=200,
# #             required_path=target_path,
# #         )
# #         if not repo_records:
# #             print(f"[skip] no python files: {repo_name}")
# #             continue

# #         all_chunks = build_repo_chunks(repo_records, mode="structure")
# #         if not all_chunks:
# #             print(f"[skip] no chunks: {repo_name}")
# #             continue

# #         gold_chunk_id = find_gold_chunk_id(
# #             all_chunks,
# #             target_path,
# #             target_line,
# #             ground_truth,
# #         )
# #         if gold_chunk_id is None:
# #             print(f"[skip] gold not matched: {repo_name} | {target_path}:{target_line}")
# #             continue

# #         retriever = BM25CodeRetriever(all_chunks)
# #         results = retriever.search(query, top_k=10)

# #         print("\n" + "#" * 100)
# #         print(f"task_id    : {ex['task_id']}")
# #         print(f"path       : {target_path}")
# #         print(f"gold_chunk : {gold_chunk_id}")
# #         if results:
# #             print(f"top1 chunk : {results[0].chunk_id}")

# #         for budget in BUDGETS:
# #             topk = pack_topk_results(results, max_context_tokens=budget)
# #             dep = pack_dependency_aware_results(
# #                 retrieval_results=results,
# #                 all_chunks=all_chunks,
# #                 max_context_tokens=budget,
# #                 max_dependency_expansions_per_chunk=2,
# #             )

# #             topk_metrics_by_budget[budget]["usable_examples"] += 1
# #             dep_metrics_by_budget[budget]["usable_examples"] += 1

# #             if packed_contains_gold_chunk(topk, gold_chunk_id):
# #                 topk_metrics_by_budget[budget]["gold_chunk_hit"] += 1
# #             if packed_contains_gold_line(topk, ground_truth):
# #                 topk_metrics_by_budget[budget]["gold_line_hit"] += 1

# #             if packed_contains_gold_chunk(dep, gold_chunk_id):
# #                 dep_metrics_by_budget[budget]["gold_chunk_hit"] += 1
# #             if packed_contains_gold_line(dep, ground_truth):
# #                 dep_metrics_by_budget[budget]["gold_line_hit"] += 1

# #             topk_metrics_by_budget[budget]["avg_tokens_sum"] += topk.total_tokens
# #             topk_metrics_by_budget[budget]["avg_chunks_sum"] += topk.num_chunks

# #             dep_metrics_by_budget[budget]["avg_tokens_sum"] += dep.total_tokens
# #             dep_metrics_by_budget[budget]["avg_chunks_sum"] += dep.num_chunks

# #             print(
# #                 f"budget={budget:>3} | "
# #                 f"topk: chunks={topk.num_chunks}, tokens={topk.total_tokens}, "
# #                 f"gold_chunk={packed_contains_gold_chunk(topk, gold_chunk_id)}, "
# #                 f"gold_line={packed_contains_gold_line(topk, ground_truth)} || "
# #                 f"dep: chunks={dep.num_chunks}, tokens={dep.total_tokens}, "
# #                 f"gold_chunk={packed_contains_gold_chunk(dep, gold_chunk_id)}, "
# #                 f"gold_line={packed_contains_gold_line(dep, ground_truth)}"
# #             )

# #     for budget in BUDGETS:
# #         print_summary(
# #             f"Top-k packing @ budget={budget}",
# #             finalize_metrics(topk_metrics_by_budget[budget]),
# #         )
# #         print_summary(
# #             f"Dependency-aware packing @ budget={budget}",
# #             finalize_metrics(dep_metrics_by_budget[budget]),
# #         )


# # if __name__ == "__main__":
# #     main()


# from __future__ import annotations

# from typing import Dict

# from src.data.load_repoeval import load_repoeval_file
# from src.packing.topk_packer import pack_topk_results
# from src.packing.dependency_packer import pack_dependency_aware_results
# from src.packing.neighbor_packer import pack_neighbor_aware_results
# from src.retrieval.bm25_retriever import BM25CodeRetriever
# from src.chunking.structure_chunker import chunk_record_structure_aware

# from scripts.run_repoeval_first_experiment import (
#     REPOEVAL_FILE,
#     find_repo_dir,
#     load_python_files_from_repo,
#     find_gold_chunk_id,
# )

# BUDGETS = [80, 120, 200]


# def build_structure_chunks(repo_records):
#     all_chunks = []
#     for record in repo_records:
#         if not record["code"].strip():
#             continue

#         chunks = chunk_record_structure_aware(
#             record,
#             code_field="code",
#             repo_field="repo",
#             path_field="path",
#             language_field="language",
#             fallback_to_fixed=True,
#             fixed_chunk_size_tokens=120,
#             fixed_chunk_overlap_tokens=20,
#             max_chunk_tokens=140,
#             overlap_tokens=20,
#         )
#         all_chunks.extend(chunks)
#     return all_chunks


# def packed_contains_gold_chunk(packed, gold_chunk_id: str) -> bool:
#     return any(c.chunk_id == gold_chunk_id for c in packed.packed_chunks)


# def packed_contains_gold_line(packed, ground_truth: str) -> bool:
#     gt = (ground_truth or "").strip()
#     if not gt:
#         return False

#     packed_lines = [line.strip() for line in packed.context_text.splitlines()]
#     return gt in packed_lines


# def init_metrics() -> Dict[str, float]:
#     return {
#         "usable_examples": 0,
#         "gold_chunk_hit": 0,
#         "gold_line_hit": 0,
#         "avg_tokens_sum": 0.0,
#         "avg_chunks_sum": 0.0,
#     }


# def finalize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
#     n = metrics["usable_examples"]
#     if n == 0:
#         return {
#             "usable_examples": 0,
#             "GoldChunkPackedRate": 0.0,
#             "GoldLinePackedRate": 0.0,
#             "AvgPackedTokens": 0.0,
#             "AvgPackedChunks": 0.0,
#         }

#     return {
#         "usable_examples": int(n),
#         "GoldChunkPackedRate": metrics["gold_chunk_hit"] / n,
#         "GoldLinePackedRate": metrics["gold_line_hit"] / n,
#         "AvgPackedTokens": metrics["avg_tokens_sum"] / n,
#         "AvgPackedChunks": metrics["avg_chunks_sum"] / n,
#     }


# def print_summary(title: str, metrics: Dict[str, float]) -> None:
#     print("\n" + "=" * 100)
#     print(title)
#     print("-" * 100)
#     for k, v in metrics.items():
#         if isinstance(v, float):
#             print(f"{k:20s}: {v:.4f}")
#         else:
#             print(f"{k:20s}: {v}")


# def main():
#     examples = load_repoeval_file(REPOEVAL_FILE, max_rows=5)

#     topk_metrics_by_budget = {b: init_metrics() for b in BUDGETS}
#     dep_metrics_by_budget = {b: init_metrics() for b in BUDGETS}
#     neighbor_metrics_by_budget = {b: init_metrics() for b in BUDGETS}

#     for ex in examples:
#         repo_name = ex["repo"]
#         target_path = ex["path"]
#         target_line = ex["line_no"]
#         query = ex["query"]
#         ground_truth = ex["ground_truth"]

#         repo_dir = find_repo_dir(repo_name)
#         if repo_dir is None:
#             print(f"[skip] repo not found: {repo_name}")
#             continue

#         repo_records = load_python_files_from_repo(
#             repo_dir,
#             max_files=200,
#             required_path=target_path,
#         )
#         if not repo_records:
#             print(f"[skip] no python files: {repo_name}")
#             continue

#         all_chunks = build_structure_chunks(repo_records)
#         if not all_chunks:
#             print(f"[skip] no chunks: {repo_name}")
#             continue

#         gold_chunk_id = find_gold_chunk_id(
#             all_chunks,
#             target_path,
#             target_line,
#             ground_truth,
#         )
#         if gold_chunk_id is None:
#             print(f"[skip] gold not matched: {repo_name} | {target_path}:{target_line}")
#             continue

#         retriever = BM25CodeRetriever(all_chunks)
#         results = retriever.search(query, top_k=10)

#         print("\n" + "#" * 100)
#         print(f"task_id    : {ex['task_id']}")
#         print(f"path       : {target_path}")
#         print(f"gold_chunk : {gold_chunk_id}")
#         if results:
#             print(f"top1 chunk : {results[0].chunk_id}")

#         for budget in BUDGETS:
#             topk = pack_topk_results(
#                 retrieval_results=results,
#                 max_context_tokens=budget,
#             )

#             dep = pack_dependency_aware_results(
#                 retrieval_results=results,
#                 all_chunks=all_chunks,
#                 max_context_tokens=budget,
#                 max_dependency_expansions_per_chunk=2,
#             )

#             neighbor = pack_neighbor_aware_results(
#                 retrieval_results=results,
#                 all_chunks=all_chunks,
#                 max_context_tokens=budget,
#                 max_neighbors_per_side=1,
#             )

#             # top-k metrics
#             topk_metrics_by_budget[budget]["usable_examples"] += 1
#             if packed_contains_gold_chunk(topk, gold_chunk_id):
#                 topk_metrics_by_budget[budget]["gold_chunk_hit"] += 1
#             if packed_contains_gold_line(topk, ground_truth):
#                 topk_metrics_by_budget[budget]["gold_line_hit"] += 1
#             topk_metrics_by_budget[budget]["avg_tokens_sum"] += topk.total_tokens
#             topk_metrics_by_budget[budget]["avg_chunks_sum"] += topk.num_chunks

#             # dependency-aware metrics
#             dep_metrics_by_budget[budget]["usable_examples"] += 1
#             if packed_contains_gold_chunk(dep, gold_chunk_id):
#                 dep_metrics_by_budget[budget]["gold_chunk_hit"] += 1
#             if packed_contains_gold_line(dep, ground_truth):
#                 dep_metrics_by_budget[budget]["gold_line_hit"] += 1
#             dep_metrics_by_budget[budget]["avg_tokens_sum"] += dep.total_tokens
#             dep_metrics_by_budget[budget]["avg_chunks_sum"] += dep.num_chunks

#             # neighbor-aware metrics
#             neighbor_metrics_by_budget[budget]["usable_examples"] += 1
#             if packed_contains_gold_chunk(neighbor, gold_chunk_id):
#                 neighbor_metrics_by_budget[budget]["gold_chunk_hit"] += 1
#             if packed_contains_gold_line(neighbor, ground_truth):
#                 neighbor_metrics_by_budget[budget]["gold_line_hit"] += 1
#             neighbor_metrics_by_budget[budget]["avg_tokens_sum"] += neighbor.total_tokens
#             neighbor_metrics_by_budget[budget]["avg_chunks_sum"] += neighbor.num_chunks

#             print(
#                 f"budget={budget:>3} | "
#                 f"topk: chunks={topk.num_chunks}, tokens={topk.total_tokens}, "
#                 f"gold_chunk={packed_contains_gold_chunk(topk, gold_chunk_id)}, "
#                 f"gold_line={packed_contains_gold_line(topk, ground_truth)} || "
#                 f"dep: chunks={dep.num_chunks}, tokens={dep.total_tokens}, "
#                 f"gold_chunk={packed_contains_gold_chunk(dep, gold_chunk_id)}, "
#                 f"gold_line={packed_contains_gold_line(dep, ground_truth)} || "
#                 f"neighbor: chunks={neighbor.num_chunks}, tokens={neighbor.total_tokens}, "
#                 f"gold_chunk={packed_contains_gold_chunk(neighbor, gold_chunk_id)}, "
#                 f"gold_line={packed_contains_gold_line(neighbor, ground_truth)}"
#             )

#     for budget in BUDGETS:
#         print_summary(
#             f"Top-k packing @ budget={budget}",
#             finalize_metrics(topk_metrics_by_budget[budget]),
#         )
#         print_summary(
#             f"Dependency-aware packing @ budget={budget}",
#             finalize_metrics(dep_metrics_by_budget[budget]),
#         )
#         print_summary(
#             f"Neighbor-aware packing @ budget={budget}",
#             finalize_metrics(neighbor_metrics_by_budget[budget]),
#         )


# if __name__ == "__main__":
#     main()


from __future__ import annotations

from typing import Dict

from src.data.load_repoeval import load_repoeval_file
from src.packing.topk_packer import pack_topk_results
from src.packing.fixed_neighbor_packer import pack_fixed_neighbor_results
from src.retrieval.bm25_retriever import BM25CodeRetriever
from src.chunking.fixed_chunker import chunk_record_fixed

from scripts.run_repoeval_first_experiment import (
    REPOEVAL_FILE,
    find_repo_dir,
    load_python_files_from_repo,
    find_gold_chunk_id,
)

BUDGETS = [80, 120, 200]


def build_fixed_chunks(repo_records):
    all_chunks = []
    for record in repo_records:
        if not record["code"].strip():
            continue

        chunks = chunk_record_fixed(
            record,
            code_field="code",
            repo_field="repo",
            path_field="path",
            language_field="language",
            chunk_size_tokens=120,
            chunk_overlap_tokens=20,
        )
        all_chunks.extend(chunks)
    return all_chunks


def packed_contains_gold_chunk(packed, gold_chunk_id: str) -> bool:
    return any(c.chunk_id == gold_chunk_id for c in packed.packed_chunks)


def packed_contains_gold_line(packed, ground_truth: str) -> bool:
    gt = (ground_truth or "").strip()
    if not gt:
        return False

    packed_lines = [line.strip() for line in packed.context_text.splitlines()]
    return gt in packed_lines


def init_metrics() -> Dict[str, float]:
    return {
        "usable_examples": 0,
        "gold_chunk_hit": 0,
        "gold_line_hit": 0,
        "avg_tokens_sum": 0.0,
        "avg_chunks_sum": 0.0,
    }


def finalize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    n = metrics["usable_examples"]
    if n == 0:
        return {
            "usable_examples": 0,
            "GoldChunkPackedRate": 0.0,
            "GoldLinePackedRate": 0.0,
            "AvgPackedTokens": 0.0,
            "AvgPackedChunks": 0.0,
        }

    return {
        "usable_examples": int(n),
        "GoldChunkPackedRate": metrics["gold_chunk_hit"] / n,
        "GoldLinePackedRate": metrics["gold_line_hit"] / n,
        "AvgPackedTokens": metrics["avg_tokens_sum"] / n,
        "AvgPackedChunks": metrics["avg_chunks_sum"] / n,
    }


def print_summary(title: str, metrics: Dict[str, float]) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("-" * 100)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:20s}: {v:.4f}")
        else:
            print(f"{k:20s}: {v}")


def main():
    examples = load_repoeval_file(REPOEVAL_FILE, max_rows=20)

    topk_metrics_by_budget = {b: init_metrics() for b in BUDGETS}
    neighbor_metrics_by_budget = {b: init_metrics() for b in BUDGETS}

    for ex in examples:
        repo_name = ex["repo"]
        target_path = ex["path"]
        target_line = ex["line_no"]
        query = ex["query"]
        ground_truth = ex["ground_truth"]

        repo_dir = find_repo_dir(repo_name)
        if repo_dir is None:
            print(f"[skip] repo not found: {repo_name}")
            continue

        repo_records = load_python_files_from_repo(
            repo_dir,
            max_files=200,
            required_path=target_path,
        )
        if not repo_records:
            print(f"[skip] no python files: {repo_name}")
            continue

        all_chunks = build_fixed_chunks(repo_records)
        if not all_chunks:
            print(f"[skip] no chunks: {repo_name}")
            continue

        gold_chunk_id = find_gold_chunk_id(
            all_chunks,
            target_path,
            target_line,
            ground_truth,
        )
        if gold_chunk_id is None:
            print(f"[skip] gold not matched: {repo_name} | {target_path}:{target_line}")
            continue

        retriever = BM25CodeRetriever(all_chunks)
        results = retriever.search(query, top_k=10)

        print("\n" + "#" * 100)
        print(f"task_id    : {ex['task_id']}")
        print(f"path       : {target_path}")
        print(f"gold_chunk : {gold_chunk_id}")
        if results:
            print(f"top1 chunk : {results[0].chunk_id}")

        for budget in BUDGETS:
            topk = pack_topk_results(
                retrieval_results=results,
                max_context_tokens=budget,
            )

            neighbor = pack_fixed_neighbor_results(
                retrieval_results=results,
                all_chunks=all_chunks,
                max_context_tokens=budget,
                max_neighbors_per_side=1,
            )

            topk_metrics_by_budget[budget]["usable_examples"] += 1
            if packed_contains_gold_chunk(topk, gold_chunk_id):
                topk_metrics_by_budget[budget]["gold_chunk_hit"] += 1
            if packed_contains_gold_line(topk, ground_truth):
                topk_metrics_by_budget[budget]["gold_line_hit"] += 1
            topk_metrics_by_budget[budget]["avg_tokens_sum"] += topk.total_tokens
            topk_metrics_by_budget[budget]["avg_chunks_sum"] += topk.num_chunks

            neighbor_metrics_by_budget[budget]["usable_examples"] += 1
            if packed_contains_gold_chunk(neighbor, gold_chunk_id):
                neighbor_metrics_by_budget[budget]["gold_chunk_hit"] += 1
            if packed_contains_gold_line(neighbor, ground_truth):
                neighbor_metrics_by_budget[budget]["gold_line_hit"] += 1
            neighbor_metrics_by_budget[budget]["avg_tokens_sum"] += neighbor.total_tokens
            neighbor_metrics_by_budget[budget]["avg_chunks_sum"] += neighbor.num_chunks

            print(
                f"budget={budget:>3} | "
                f"topk: chunks={topk.num_chunks}, tokens={topk.total_tokens}, "
                f"gold_chunk={packed_contains_gold_chunk(topk, gold_chunk_id)}, "
                f"gold_line={packed_contains_gold_line(topk, ground_truth)} || "
                f"fixed-neighbor: chunks={neighbor.num_chunks}, tokens={neighbor.total_tokens}, "
                f"gold_chunk={packed_contains_gold_chunk(neighbor, gold_chunk_id)}, "
                f"gold_line={packed_contains_gold_line(neighbor, ground_truth)}"
            )

    for budget in BUDGETS:
        print_summary(
            f"Top-k packing on fixed chunks @ budget={budget}",
            finalize_metrics(topk_metrics_by_budget[budget]),
        )
        print_summary(
            f"Fixed-neighbor packing @ budget={budget}",
            finalize_metrics(neighbor_metrics_by_budget[budget]),
        )


if __name__ == "__main__":
    main()