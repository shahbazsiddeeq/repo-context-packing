from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.data.load_repoeval import load_repoeval_file
from src.chunking.fixed_chunker import chunk_record_fixed
from src.retrieval.bm25_retriever import BM25CodeRetriever
from src.retrieval.locality_reranker import rerank_with_locality

from scripts.run_repoeval_first_experiment import (
    REPOEVAL_FILE,
    find_repo_dir,
    load_python_files_from_repo,
)

# -----------------------------
# Safer attention backend for mixed/older GPU environments.
# This avoids FlashAttention / SDP paths that can fail on non-Ampere GPUs.
# -----------------------------
if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

# -----------------------------
# Main experiment settings
# -----------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "bigcode/starcoder2-7b")

max_rows_env = os.getenv("MAX_ROWS", "5").strip().lower()
if max_rows_env in {"all", "none", ""}:
    MAX_ROWS: Optional[int] = None
else:
    MAX_ROWS = int(max_rows_env)

TOP_K = int(os.getenv("TOP_K", "10"))
MAX_FILES_PER_REPO = int(os.getenv("MAX_FILES_PER_REPO", "1000"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "120"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))

SAME_FILE_BONUS = float(os.getenv("SAME_FILE_BONUS", "150.0"))
LINE_DISTANCE_WEIGHT = float(os.getenv("LINE_DISTANCE_WEIGHT", "0.2"))

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "32"))
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "16000"))
DEBUG_FIRST_N = int(os.getenv("DEBUG_FIRST_N", "5"))

# OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/completion_step1_smoketest"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/completion_step1_1600"))
PER_EXAMPLE_CSV = OUTPUT_DIR / "repoeval_completion_step1_per_example.csv"
SUMMARY_TXT = OUTPUT_DIR / "repoeval_completion_step1_summary.txt"


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to save for {path}")

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_fixed_chunks(repo_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []
    for record in repo_records:
        if not record["code"].strip():
            continue

        chunks = chunk_record_fixed(
            record,
            code_field="code",
            repo_field="repo",
            path_field="path",
            language_field="language",
            chunk_size_tokens=CHUNK_SIZE,
            chunk_overlap_tokens=CHUNK_OVERLAP,
        )
        all_chunks.extend(chunks)
    return all_chunks


def normalize_line(text: str) -> str:
    return text.strip()


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_line(pred) == normalize_line(gold))


# def extract_next_line(generated_text: str) -> str:
#     text = generated_text.replace("\r\n", "\n").replace("\r", "\n")
#     line = text.split("\n", 1)[0]
#     return line.rstrip()

def extract_next_line(generated_text: str) -> str:
    text = generated_text.replace("\r\n", "\n").replace("\r", "\n")

    # very important: drop only leading newlines, keep indentation
    text = text.lstrip("\n")

    if not text:
        return ""

    line = text.split("\n", 1)[0]
    return line.rstrip()

# def build_prompt_only(target_path: str, prompt: str) -> str:
#     return (
#         "### Current File Path\n"
#         f"{target_path}\n\n"
#         "### Current Code Prefix\n"
#         f"{prompt}\n"
#     )

def build_prompt_only(target_path: str, prompt: str) -> str:
    return prompt


# def build_prompt_with_context(target_path: str, prompt: str, context_text: str) -> str:
#     return (
#         "### Repository Context\n"
#         f"{context_text}\n\n"
#         "### Current File Path\n"
#         f"{target_path}\n\n"
#         "### Current Code Prefix\n"
#         f"{prompt}\n"
#     )

def build_prompt_with_context(target_path: str, prompt: str, context_text: str) -> str:
    commented_context = "\n".join(
        f"# {line}" if line.strip() else "#"
        for line in context_text.splitlines()
    )
    return (
        f"# Retrieved repository context\n"
        f"# Target file: {target_path}\n"
        f"{commented_context}\n\n"
        f"{prompt}"
    )


@torch.no_grad()
def generate_next_line(
    model,
    tokenizer,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # print("RAW_GENERATED:", repr(generated_text[:200]))

    return extract_next_line(generated_text)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # custom sp
    tokenizer.truncation_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = choose_device()

    if device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
        )
    elif device.type == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            attn_implementation="eager",
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            attn_implementation="eager",
        ).to(device)

    model.eval()
    return model, tokenizer, device


def main():
    examples = load_repoeval_file(REPOEVAL_FILE, max_rows=MAX_ROWS)
    print(f"Loaded examples: {len(examples)}")

    model, tokenizer, device = load_model_and_tokenizer(MODEL_NAME)
    print(f"Loaded model: {MODEL_NAME}")
    print(f"Using device: {device}")

    rows: List[Dict[str, Any]] = []

    used_examples = 0
    skipped_repo = 0
    skipped_files = 0
    skipped_chunks = 0
    skipped_retrieval = 0

    em_prompt_only_sum = 0
    em_bm25_sum = 0
    em_locality_sum = 0

    for idx, ex in enumerate(examples, start=1):
        repo_name = ex["repo"]
        target_path = ex["path"]
        retrieval_query = ex["query"]
        generation_prefix = ex["prompt"]
        gold_line = ex["ground_truth"]
        target_line = ex["line_no"]

        print(f"[{idx}/{len(examples)}] {ex['task_id']}")

        repo_dir = find_repo_dir(repo_name)
        if repo_dir is None:
            skipped_repo += 1
            continue

        repo_records = load_python_files_from_repo(
            repo_dir,
            max_files=MAX_FILES_PER_REPO,
            required_path=target_path,
        )
        if not repo_records:
            skipped_files += 1
            continue

        all_chunks = build_fixed_chunks(repo_records)
        if not all_chunks:
            skipped_chunks += 1
            continue

        retriever = BM25CodeRetriever(all_chunks)
        bm25_results = retriever.search(retrieval_query, top_k=TOP_K)
        if not bm25_results:
            skipped_retrieval += 1
            continue

        locality_results = rerank_with_locality(
            bm25_results,
            current_path=target_path,
            current_line=target_line + 1,
            same_file_bonus=SAME_FILE_BONUS,
            line_distance_weight=LINE_DISTANCE_WEIGHT,
        )
        if not locality_results:
            skipped_retrieval += 1
            continue

        bm25_top1 = bm25_results[0]
        locality_top1 = locality_results[0]

        prompt_only_text = build_prompt_only(target_path, generation_prefix)
        bm25_prompt_text = build_prompt_with_context(target_path, generation_prefix, bm25_top1.text)
        locality_prompt_text = build_prompt_with_context(target_path, generation_prefix, locality_top1.text)

        pred_prompt_only = generate_next_line(model, tokenizer, prompt_only_text, device)
        pred_bm25 = generate_next_line(model, tokenizer, bm25_prompt_text, device)
        pred_locality = generate_next_line(model, tokenizer, locality_prompt_text, device)

        if idx <= DEBUG_FIRST_N:
            print("GOLD:", repr(gold_line))
            print("PROMPT_ONLY:", repr(pred_prompt_only))
            print("BM25:", repr(pred_bm25))
            print("LOCALITY:", repr(pred_locality))
            print("BM25_TOP1_PATH:", repr(bm25_top1.metadata.get("path", "")))
            print("LOCALITY_TOP1_PATH:", repr(locality_top1.metadata.get("path", "")))
            print("-" * 80)

        em_prompt_only = exact_match(pred_prompt_only, gold_line)
        em_bm25 = exact_match(pred_bm25, gold_line)
        em_locality = exact_match(pred_locality, gold_line)

        em_prompt_only_sum += em_prompt_only
        em_bm25_sum += em_bm25
        em_locality_sum += em_locality

        used_examples += 1

        rows.append(
            {
                "task_id": ex["task_id"],
                "repo": repo_name,
                "path": target_path,
                "line_no": target_line,
                "gold_line": gold_line,
                "retrieval_query": retrieval_query,
                "generation_prefix_tail": generation_prefix[-500:],
                "pred_prompt_only": pred_prompt_only,
                "pred_bm25": pred_bm25,
                "pred_locality": pred_locality,
                "em_prompt_only": em_prompt_only,
                "em_bm25": em_bm25,
                "em_locality": em_locality,
                "bm25_top1_chunk": bm25_top1.chunk_id,
                "bm25_top1_path": bm25_top1.metadata.get("path", ""),
                "bm25_top1_start": bm25_top1.metadata.get("start_line", -1),
                "bm25_top1_end": bm25_top1.metadata.get("end_line", -1),
                "locality_top1_chunk": locality_top1.chunk_id,
                "locality_top1_path": locality_top1.metadata.get("path", ""),
                "locality_top1_start": locality_top1.metadata.get("start_line", -1),
                "locality_top1_end": locality_top1.metadata.get("end_line", -1),
            }
        )

    if used_examples == 0:
        raise RuntimeError("No usable examples were evaluated.")

    save_csv(PER_EXAMPLE_CSV, rows)

    em_prompt_only = em_prompt_only_sum / used_examples
    em_bm25 = em_bm25_sum / used_examples
    em_locality = em_locality_sum / used_examples

    report = []
    report.append("RepoEval completion step-1 experiment")
    report.append("")
    report.append(f"Model: {MODEL_NAME}")
    report.append(f"Examples requested: {len(examples)}")
    report.append(f"Usable examples: {used_examples}")
    report.append(f"Skipped repo: {skipped_repo}")
    report.append(f"Skipped files: {skipped_files}")
    report.append(f"Skipped chunks: {skipped_chunks}")
    report.append(f"Skipped retrieval: {skipped_retrieval}")
    report.append("")
    report.append(f"Prompt-only EM: {em_prompt_only:.4f}")
    report.append(f"BM25 top-1 context EM: {em_bm25:.4f}")
    report.append(f"Locality top-1 context EM: {em_locality:.4f}")
    report.append("")
    report.append(f"Delta (BM25 - Prompt-only): {em_bm25 - em_prompt_only:.4f}")
    report.append(f"Delta (Locality - BM25): {em_locality - em_bm25:.4f}")
    report.append(f"Delta (Locality - Prompt-only): {em_locality - em_prompt_only:.4f}")
    report.append("")
    report.append(f"Chunk size: {CHUNK_SIZE}")
    report.append(f"Chunk overlap: {CHUNK_OVERLAP}")
    report.append(f"Top-k retrieval: {TOP_K}")
    report.append(f"Same-file bonus: {SAME_FILE_BONUS}")
    report.append(f"Line-distance weight: {LINE_DISTANCE_WEIGHT}")
    report.append(f"Max new tokens: {MAX_NEW_TOKENS}")
    report.append(f"Max input tokens: {MAX_INPUT_TOKENS}")
    report.append(f"Debug first N: {DEBUG_FIRST_N}")
    report.append(f"Output dir: {OUTPUT_DIR}")

    save_text(SUMMARY_TXT, "\n".join(report))

    print(f"Saved per-example CSV to: {PER_EXAMPLE_CSV}")
    print(f"Saved summary to: {SUMMARY_TXT}")
    print()
    print("\n".join(report))


if __name__ == "__main__":
    main()
