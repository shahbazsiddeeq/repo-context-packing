from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Any, List, Set

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast

from src.data.load_repoeval import load_repoeval_file
from src.chunking.fixed_chunker import chunk_record_fixed
from src.retrieval.bm25_retriever import BM25CodeRetriever
from src.retrieval.locality_reranker import rerank_with_locality

from scripts.run_repoeval_first_experiment import (
    REPOEVAL_FILE,
    find_repo_dir,
    load_python_files_from_repo,
)

TOP_K = 10
MAX_FILES_PER_REPO = 1000
CHUNK_SIZE = 120
CHUNK_OVERLAP = 20
SAME_FILE_BONUS = 150.0
LINE_DISTANCE_WEIGHT = 0.2
MAX_NEW_TOKENS = 32


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to save for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path: Path):
    """Load existing rows from CSV checkpoint. Returns (rows, done_task_ids)."""
    if not path.exists():
        return [], set()
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # restore int columns
            for col in ("line_no", "em_prompt_only", "em_bm25", "em_locality",
                        "bm25_top1_start", "bm25_top1_end",
                        "locality_top1_start", "locality_top1_end"):
                if col in row:
                    row[col] = int(row[col])
            rows.append(row)
    done_ids: Set[str] = {r["task_id"] for r in rows}
    return rows, done_ids


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


def extract_next_line(generated_text: str) -> str:
    text = generated_text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.lstrip("\n")
    return text.split("\n", 1)[0].rstrip()


def build_prompt_only_instruction(target_path: str, prompt: str) -> str:
    return (
        "### Current File Path\n"
        f"{target_path}\n\n"
        "### Current Code Prefix\n"
        f"{prompt}\n\n"
        "### Task\n"
        "Predict the next line of code only.\n"
    )


def build_prompt_with_context_instruction(target_path: str, prompt: str, context_text: str) -> str:
    return (
        "### Repository Context\n"
        f"{context_text}\n\n"
        "### Current File Path\n"
        f"{target_path}\n\n"
        "### Current Code Prefix\n"
        f"{prompt}\n\n"
        "### Task\n"
        "Predict the next line of code only.\n"
    )


def build_prompt_only_raw(target_path: str, prompt: str) -> str:
    # Plain code prefix — for base models that do text continuation (e.g. CodeLlama, DeepSeek base)
    return f"# {target_path}\n{prompt}"


def build_prompt_with_context_raw(target_path: str, prompt: str, context_text: str) -> str:
    return f"{context_text}\n\n# {target_path}\n{prompt}"


def build_prompt_only(target_path: str, prompt: str, style: str) -> str:
    if style == "raw":
        return build_prompt_only_raw(target_path, prompt)
    return build_prompt_only_instruction(target_path, prompt)


def build_prompt_with_context(target_path: str, prompt: str, context_text: str, style: str) -> str:
    if style == "raw":
        return build_prompt_with_context_raw(target_path, prompt, context_text)
    return build_prompt_with_context_instruction(target_path, prompt, context_text)


@torch.no_grad()
def generate_next_line(
    model,
    tokenizer,
    prompt_text: str,
    device: torch.device,
    max_input_tokens: int,
) -> str:
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_len:]
    return extract_next_line(tokenizer.decode(generated_ids, skip_special_tokens=True))


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(model_name: str):
    # Some models (e.g. DeepSeek-Coder) ship only tokenizer.json (no tokenizer.model).
    # AutoTokenizer falls back to the slow LlamaTokenizer which strips all whitespace.
    # PreTrainedTokenizerFast uses the Rust tokenizers backend and handles whitespace correctly.
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = choose_device()

    if device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
        )
    elif device.type == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        ).to(device)

    model.eval()
    return model, tokenizer, device


def parse_args():
    p = argparse.ArgumentParser(description="RepoEval completion experiment (any model)")
    p.add_argument("--model_name", required=True, help="HuggingFace model ID")
    p.add_argument("--max_rows", type=int, default=1600, help="Number of examples to evaluate")
    p.add_argument("--output_dir", required=True, help="Directory for CSV and summary outputs")
    p.add_argument(
        "--max_input_tokens",
        type=int,
        default=4096,
        help="Max tokenizer truncation length (use 4096 for CodeLlama/DeepSeek/CodeQwen)",
    )
    p.add_argument(
        "--checkpoint_every",
        type=int,
        default=50,
        help="Save checkpoint CSV every N new examples (default 50)",
    )
    p.add_argument(
        "--prompt_style",
        choices=["instruction", "raw"],
        default="instruction",
        help="'instruction' = markdown headers (StarCoder2); 'raw' = plain code prefix (CodeLlama/DeepSeek base)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name
    max_rows = args.max_rows
    output_dir = Path(args.output_dir)
    max_input_tokens = args.max_input_tokens
    checkpoint_every = args.checkpoint_every
    prompt_style = args.prompt_style

    stem = output_dir.name
    per_example_csv = output_dir / f"{stem}_per_example.csv"
    summary_txt = output_dir / f"{stem}_summary.txt"

    examples = load_repoeval_file(REPOEVAL_FILE, max_rows=max_rows)
    print(f"Loaded examples: {len(examples)}")

    # Resume from checkpoint if it exists
    rows, done_ids = load_checkpoint(per_example_csv)
    if done_ids:
        print(f"Resuming from checkpoint: {len(done_ids)} examples already done, skipping them.")

    # Recompute running totals from loaded rows
    em_prompt_only_sum = sum(int(r["em_prompt_only"]) for r in rows)
    em_bm25_sum = sum(int(r["em_bm25"]) for r in rows)
    em_locality_sum = sum(int(r["em_locality"]) for r in rows)
    used_examples = len(rows)

    skipped_repo = 0
    skipped_files = 0
    skipped_chunks = 0
    skipped_retrieval = 0
    new_since_checkpoint = 0

    model, tokenizer, device = load_model_and_tokenizer(model_name)
    print(f"Loaded model: {model_name}")
    print(f"Using device: {device}")
    print(f"Max input tokens: {max_input_tokens}")
    print(f"Checkpoint every: {checkpoint_every} examples")
    print(f"Prompt style: {prompt_style}")

    for idx, ex in enumerate(examples, start=1):
        task_id = ex["task_id"]

        if task_id in done_ids:
            continue

        repo_name = ex["repo"]
        target_path = ex["path"]
        retrieval_query = ex["query"]
        generation_prefix = ex.get("prompt", ex["query"])
        gold_line = ex["ground_truth"]
        target_line = ex["line_no"]

        print(f"[{idx}/{len(examples)}] {task_id}")

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

        bm25_top1 = bm25_results[0]
        locality_top1 = locality_results[0]

        prompt_only_text = build_prompt_only(target_path, generation_prefix, prompt_style)
        bm25_prompt_text = build_prompt_with_context(target_path, retrieval_query, bm25_top1.text, prompt_style)
        locality_prompt_text = build_prompt_with_context(target_path, retrieval_query, locality_top1.text, prompt_style)

        pred_prompt_only = generate_next_line(model, tokenizer, prompt_only_text, device, max_input_tokens)
        pred_bm25 = generate_next_line(model, tokenizer, bm25_prompt_text, device, max_input_tokens)
        pred_locality = generate_next_line(model, tokenizer, locality_prompt_text, device, max_input_tokens)

        if used_examples < 5:
            print("GOLD:", repr(gold_line))
            print("PROMPT_ONLY:", repr(pred_prompt_only))
            print("BM25:", repr(pred_bm25))
            print("LOCALITY:", repr(pred_locality))
            print("-" * 80)

        em_po = exact_match(pred_prompt_only, gold_line)
        em_b = exact_match(pred_bm25, gold_line)
        em_l = exact_match(pred_locality, gold_line)

        em_prompt_only_sum += em_po
        em_bm25_sum += em_b
        em_locality_sum += em_l
        used_examples += 1
        new_since_checkpoint += 1

        rows.append({
            "task_id": task_id,
            "repo": repo_name,
            "path": target_path,
            "line_no": target_line,
            "gold_line": gold_line,
            "pred_prompt_only": pred_prompt_only,
            "pred_bm25": pred_bm25,
            "pred_locality": pred_locality,
            "em_prompt_only": em_po,
            "em_bm25": em_b,
            "em_locality": em_l,
            "bm25_top1_chunk": bm25_top1.chunk_id,
            "bm25_top1_path": bm25_top1.metadata.get("path", ""),
            "bm25_top1_start": bm25_top1.metadata.get("start_line", -1),
            "bm25_top1_end": bm25_top1.metadata.get("end_line", -1),
            "locality_top1_chunk": locality_top1.chunk_id,
            "locality_top1_path": locality_top1.metadata.get("path", ""),
            "locality_top1_start": locality_top1.metadata.get("start_line", -1),
            "locality_top1_end": locality_top1.metadata.get("end_line", -1),
        })

        if new_since_checkpoint >= checkpoint_every:
            save_csv(per_example_csv, rows)
            print(f"  [checkpoint] saved {used_examples} rows to {per_example_csv}")
            new_since_checkpoint = 0

    if used_examples == 0:
        raise RuntimeError("No usable examples were evaluated.")

    save_csv(per_example_csv, rows)

    em_po_final = em_prompt_only_sum / used_examples
    em_b_final = em_bm25_sum / used_examples
    em_l_final = em_locality_sum / used_examples

    report = [
        "RepoEval completion experiment",
        "",
        f"Model: {model_name}",
        f"Examples requested: {len(examples)}",
        f"Usable examples: {used_examples}",
        f"Skipped repo: {skipped_repo}",
        f"Skipped files: {skipped_files}",
        f"Skipped chunks: {skipped_chunks}",
        f"Skipped retrieval: {skipped_retrieval}",
        "",
        f"Prompt-only EM: {em_po_final:.4f}",
        f"BM25 top-1 context EM: {em_b_final:.4f}",
        f"Locality top-1 context EM: {em_l_final:.4f}",
        "",
        f"Delta (BM25 - Prompt-only): {em_b_final - em_po_final:.4f}",
        f"Delta (Locality - BM25): {em_l_final - em_b_final:.4f}",
        f"Delta (Locality - Prompt-only): {em_l_final - em_po_final:.4f}",
        "",
        f"Chunk size: {CHUNK_SIZE}",
        f"Chunk overlap: {CHUNK_OVERLAP}",
        f"Top-k retrieval: {TOP_K}",
        f"Same-file bonus: {SAME_FILE_BONUS}",
        f"Line-distance weight: {LINE_DISTANCE_WEIGHT}",
        f"Max input tokens: {max_input_tokens}",
        f"Prompt style: {prompt_style}",
    ]

    save_text(summary_txt, "\n".join(report))
    print(f"\nSaved per-example CSV: {per_example_csv}")
    print(f"Saved summary: {summary_txt}")
    print()
    print("\n".join(report))


if __name__ == "__main__":
    main()
