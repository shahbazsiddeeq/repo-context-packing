"""Dense+Locality end-to-end completion experiment (RQ5 extension).

Evaluates Dense and Dense+Locality retrieval with StarCoder2-7B generation
on RepoEval (1600 examples). Chunks are encoded once per repo using
UniXcoder+FAISS; the top-1 retrieved chunk drives generation.

Conditions evaluated:
  Dense          -- UniXcoder top-1 chunk as context
  Dense+Locality -- UniXcoder top-10, locality-reranked, top-1 as context

Usage (L4 GPU, ~14.5 GB VRAM total):
  screen -dmS dense_em bash -c '
    export PYTHONPATH=/home/shahbaz/repo-context-packing &&
    CUDA_VISIBLE_DEVICES=0 .venv-linux/bin/python -u \\
      scripts/run_dense_completion_model.py \\
      --model_name bigcode/starcoder2-7b \\
      --output_dir outputs/completion_dense_starcoder2_7b \\
      2>&1 | tee outputs/completion_dense_starcoder2_7b.log'

Outputs (in --output_dir):
  dense_completion_per_example.csv   per-example results
  dense_completion_summary.txt       aggregate EM + Δ
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

from src.chunking.fixed_chunker import chunk_record_fixed
from src.data.load_repoeval import load_repoeval_file
from src.retrieval.locality_reranker import rerank_with_locality

from scripts.run_repoeval_first_experiment import (
    REPOEVAL_FILE,
    find_repo_dir,
    load_python_files_from_repo,
)

# ── shared with run_repoeval_completion_model ────────────────────────────────
from scripts.run_repoeval_completion_model import (
    MAX_NEW_TOKENS,
    TOP_K,
    MAX_FILES_PER_REPO,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SAME_FILE_BONUS,
    LINE_DISTANCE_WEIGHT,
    build_fixed_chunks,
    normalize_line,
    exact_match,
    extract_next_line,
    build_prompt_with_context,
    generate_next_line,
    choose_device,
    load_model_and_tokenizer,
)

UNIXCODER_NAME = "microsoft/unixcoder-base"
BATCH_SIZE     = 64
CHECKPOINT_EVERY = 50

CSV_FIELDS = [
    "task_id", "repo", "path", "line_no", "gold_line",
    "pred_dense", "pred_dense_loc",
    "em_dense", "em_dense_loc",
    "dense_top1_path", "dense_loc_top1_path",
]


# ── UniXcoder encoder (minimal, self-contained) ──────────────────────────────
class UniXcoderEncoder:
    def __init__(self, device: torch.device):
        from pathlib import Path as _Path
        _cache = _Path.home() / ".cache/huggingface/hub"
        _snap = (
            _cache / "models--microsoft--unixcoder-base" / "snapshots"
            / "5604afdc964f6c53782a6813140ade5216b99006"
        )
        local = str(_snap) if _snap.exists() else UNIXCODER_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(local)
        self.model = AutoModel.from_pretrained(local, use_safetensors=True).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(self.device)
        out = self.model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        embs = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        embs = embs.cpu().float().numpy()
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.maximum(norms, 1e-9)

    def encode(self, texts: List[str]) -> np.ndarray:
        batches = [
            self._encode_batch(texts[i:i + BATCH_SIZE])
            for i in range(0, len(texts), BATCH_SIZE)
        ]
        return np.vstack(batches)


def build_faiss_index(embs: np.ndarray) -> faiss.IndexFlatIP:
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs.astype(np.float32))
    return idx


# ── checkpoint helpers ───────────────────────────────────────────────────────
def load_checkpoint(csv_path: Path):
    if not csv_path.exists():
        return [], set()
    rows: List[Dict] = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["line_no"]    = int(row["line_no"])
            row["em_dense"]   = int(row["em_dense"])
            row["em_dense_loc"] = int(row["em_dense_loc"])
            rows.append(row)
    return rows, {r["task_id"] for r in rows}


def save_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ── DenseResult stub so locality_reranker can consume dense results ──────────
from dataclasses import dataclass

@dataclass
class DenseResult:
    chunk_id: str
    score: float
    rank: int
    text: str
    metadata: Dict[str, Any]


def dense_search(
    index: faiss.IndexFlatIP,
    query_emb: np.ndarray,
    chunks: List[Dict],
    top_k: int = TOP_K,
) -> List[DenseResult]:
    scores, indices = index.search(
        query_emb.reshape(1, -1).astype(np.float32), top_k
    )
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if idx < 0:
            continue
        c = chunks[idx]
        results.append(DenseResult(
            chunk_id=c["chunk_id"],
            score=float(score),
            rank=rank,
            text=c["text"],
            metadata={k: c.get(k) for k in ("repo", "path", "start_line", "end_line")},
        ))
    return results


# ── arg parsing ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Dense+Locality end-to-end completion (RQ5)")
    p.add_argument("--model_name", default="bigcode/starcoder2-7b")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_rows", type=int, default=1600)
    p.add_argument("--max_input_tokens", type=int, default=4096)
    p.add_argument(
        "--prompt_style", choices=["instruction", "raw"], default="instruction",
        help="instruction for StarCoder2; raw for CodeLlama/DeepSeek/CodeQwen base",
    )
    return p.parse_args()


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_example_csv = output_dir / "dense_completion_per_example.csv"
    summary_txt     = output_dir / "dense_completion_summary.txt"

    device = choose_device()
    print(f"Device: {device}")

    # ── load UniXcoder first (small: ~0.5 GB) ────────────────────────────────
    print("\nLoading UniXcoder encoder...")
    encoder = UniXcoderEncoder(device)
    print("UniXcoder loaded.")

    # ── load generation model ────────────────────────────────────────────────
    print(f"\nLoading generation model: {args.model_name} ...")
    gen_model, gen_tokenizer, _ = load_model_and_tokenizer(args.model_name)
    print(f"Generation model loaded. Prompt style: {args.prompt_style}")

    # ── load examples and group by repo ─────────────────────────────────────
    print(f"\nLoading {args.max_rows} RepoEval examples...")
    examples = load_repoeval_file(REPOEVAL_FILE, max_rows=args.max_rows)
    print(f"Loaded: {len(examples)} examples")

    rows, done_ids = load_checkpoint(per_example_csv)
    if done_ids:
        print(f"Resuming — {len(done_ids)} already done")

    # recompute running totals from checkpoint
    em_dense_sum     = sum(r["em_dense"]     for r in rows)
    em_dense_loc_sum = sum(r["em_dense_loc"] for r in rows)
    used             = len(rows)
    new_since_ckpt   = 0
    skipped          = 0

    by_repo: Dict[str, List[Dict]] = defaultdict(list)
    for ex in examples:
        by_repo[ex["repo"]].append(ex)

    repos = sorted(by_repo.keys())
    print(f"Repos: {len(repos)}\n")

    for repo_num, repo_name in enumerate(repos, start=1):
        repo_examples = by_repo[repo_name]
        pending = [e for e in repo_examples if e["task_id"] not in done_ids]

        print(f"[repo {repo_num}/{len(repos)}] {repo_name} — "
              f"{len(pending)}/{len(repo_examples)} pending")

        if not pending:
            print("  all done, skipping")
            continue

        repo_dir = find_repo_dir(repo_name)
        if repo_dir is None:
            skipped += len(pending); print("  [skip] repo dir not found"); continue

        repo_records = load_python_files_from_repo(repo_dir, max_files=MAX_FILES_PER_REPO)
        if not repo_records:
            skipped += len(pending); print("  [skip] no python files"); continue

        all_chunks = build_fixed_chunks(repo_records)
        if not all_chunks:
            skipped += len(pending); print("  [skip] no chunks"); continue

        # ── encode all chunks for this repo once ────────────────────────────
        print(f"  chunks: {len(all_chunks)} — encoding with UniXcoder...")
        t0 = time.perf_counter()
        chunk_embs = encoder.encode([c["text"] for c in all_chunks])
        faiss_index = build_faiss_index(chunk_embs)
        t1 = time.perf_counter()
        print(f"  encoded in {t1 - t0:.1f}s")

        for ex in pending:
            task_id  = ex["task_id"]
            path     = ex["path"]
            query    = ex["query"]
            prefix   = ex.get("prompt", query)
            gold     = ex["ground_truth"]
            line_no  = ex["line_no"]

            print(f"  {task_id}")

            # Dense retrieval
            q_emb = encoder.encode([query])
            dense_results = dense_search(faiss_index, q_emb[0], all_chunks, top_k=TOP_K)
            if not dense_results:
                skipped += 1; continue

            # Dense+Locality reranking
            dense_loc_results = rerank_with_locality(
                dense_results,
                current_path=path,
                current_line=line_no + 1,
                same_file_bonus=SAME_FILE_BONUS,
                line_distance_weight=LINE_DISTANCE_WEIGHT,
            )

            top1_dense     = dense_results[0]
            top1_dense_loc = dense_loc_results[0]

            prompt_dense = build_prompt_with_context(
                path, prefix, top1_dense.text, args.prompt_style
            )
            prompt_dense_loc = build_prompt_with_context(
                path, prefix, top1_dense_loc.text, args.prompt_style
            )

            pred_dense     = generate_next_line(gen_model, gen_tokenizer, prompt_dense,     device, args.max_input_tokens)
            pred_dense_loc = generate_next_line(gen_model, gen_tokenizer, prompt_dense_loc, device, args.max_input_tokens)

            em_d   = exact_match(pred_dense,     gold)
            em_dl  = exact_match(pred_dense_loc, gold)

            em_dense_sum     += em_d
            em_dense_loc_sum += em_dl
            used             += 1
            new_since_ckpt   += 1

            rows.append({
                "task_id":          task_id,
                "repo":             repo_name,
                "path":             path,
                "line_no":          line_no,
                "gold_line":        gold,
                "pred_dense":       pred_dense,
                "pred_dense_loc":   pred_dense_loc,
                "em_dense":         em_d,
                "em_dense_loc":     em_dl,
                "dense_top1_path":     top1_dense.metadata.get("path", ""),
                "dense_loc_top1_path": top1_dense_loc.metadata.get("path", ""),
            })

            if new_since_ckpt >= CHECKPOINT_EVERY:
                save_csv(per_example_csv, rows)
                print(f"    [checkpoint] {used} rows | "
                      f"Dense EM={em_dense_sum/used:.4f} "
                      f"Dense+Loc EM={em_dense_loc_sum/used:.4f}")
                new_since_ckpt = 0

    if used == 0:
        raise RuntimeError("No usable examples processed.")

    save_csv(per_example_csv, rows)

    em_d_final  = em_dense_sum     / used
    em_dl_final = em_dense_loc_sum / used

    report_lines = [
        "Dense+Locality end-to-end completion (RQ5 extension)",
        "",
        f"Generation model:  {args.model_name}",
        f"Dense encoder:     {UNIXCODER_NAME}",
        f"Prompt style:      {args.prompt_style}",
        f"Examples requested:{args.max_rows}",
        f"Usable examples:   {used}",
        f"Skipped:           {skipped}",
        "",
        f"Dense EM:          {em_d_final*100:.2f}%  ({em_dense_sum}/{used})",
        f"Dense+Locality EM: {em_dl_final*100:.2f}%  ({em_dense_loc_sum}/{used})",
        f"Δ (Dense+Loc - Dense): {(em_dl_final - em_d_final)*100:+.2f} pp",
        "",
        "For reference (from previous runs):",
        "  BM25 EM:          78.19%  (StarCoder2-7B, 1600 examples)",
        "  BM25+Locality EM: 79.38%  (StarCoder2-7B, 1600 examples)",
        "",
        f"Chunk size:        {CHUNK_SIZE} tokens",
        f"Chunk overlap:     {CHUNK_OVERLAP} tokens",
        f"Top-k:             {TOP_K}",
        f"Same-file bonus β: {SAME_FILE_BONUS}",
        f"Line-distance λ:   {LINE_DISTANCE_WEIGHT}",
        f"Max input tokens:  {args.max_input_tokens}",
    ]

    summary_txt.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n{'='*70}")
    print("\n".join(report_lines))
    print(f"\nSaved: {per_example_csv}")
    print(f"Saved: {summary_txt}")


if __name__ == "__main__":
    main()
