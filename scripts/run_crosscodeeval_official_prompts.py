from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Safer attention backend for mixed/older GPU environments.
if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run StarCoder2-7B on official CrossCodeEval prompt JSONL files."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a CrossCodeEval JSONL file, e.g. data/python/line_completion.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for CSV and summary outputs.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=os.getenv("MODEL_NAME", "bigcode/starcoder2-7b"),
        help="Hugging Face model name.",
    )
    parser.add_argument(
        "--max_rows",
        type=str,
        default=os.getenv("MAX_ROWS", "5"),
        help="Number of examples to evaluate, or 'all'.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=int(os.getenv("MAX_NEW_TOKENS", "32")),
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=int(os.getenv("MAX_INPUT_TOKENS", "16000")),
        help="Maximum input token budget.",
    )
    parser.add_argument(
        "--debug_first_n",
        type=int,
        default=int(os.getenv("DEBUG_FIRST_N", "5")),
        help="Print debug output for the first N examples.",
    )
    parser.add_argument(
        "--use_crossfile_context",
        type=str,
        default=os.getenv("USE_CROSSFILE_CONTEXT", "auto"),
        choices=["auto", "always", "never"],
        help="Whether to prepend official cross-file context when present.",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=int(os.getenv("CHECKPOINT_EVERY", "50")),
        help="Save a checkpoint CSV every N examples (0 to disable).",
    )
    return parser.parse_args()


def maybe_int(value: str) -> Optional[int]:
    value = value.strip().lower()
    if value in {"all", "none", ""}:
        return None
    return int(value)


def save_csv(path: Path, rows: List[Dict[str, Any]], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    write_header = not append or not path.exists()
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def infer_language(input_path: str) -> str:
    p = Path(input_path)
    for part in p.parts:
        if part.lower() in {"python", "java", "typescript", "csharp"}:
            return part.lower()
    return "unknown"


def load_crosscodeeval_file(input_path: str, max_rows: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    language = infer_language(input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_rows is not None and idx >= max_rows:
                break
            if not line.strip():
                continue
            ex = json.loads(line)
            meta = ex.get("metadata", {})
            cross = ex.get("crossfile_context", {}) or {}
            rows.append(
                {
                    "task_id": meta.get("task_id", f"{language}/{idx}"),
                    "repo": meta.get("repository", ""),
                    "path": meta.get("file", ""),
                    "language": language,
                    "prompt": ex.get("prompt", ""),
                    "ground_truth": ex.get("groundtruth", ""),
                    "right_context": ex.get("right_context", ""),
                    "line_no": meta.get("groundtruth_start_lineno", -1),
                    "context_start_lineno": meta.get("context_start_lineno", -1),
                    "right_context_start_lineno": meta.get("right_context_start_lineno", -1),
                    "crossfile_context_text": cross.get("text", ""),
                    "crossfile_context_list": cross.get("list", []),
                }
            )
    return rows


def normalize_line(text: str) -> str:
    return text.strip()


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_line(pred) == normalize_line(gold))


def extract_next_line(generated_text: str) -> str:
    text = generated_text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.lstrip("\n")
    if not text:
        return ""
    return text.split("\n", 1)[0].rstrip()


def build_prompt_only(prompt: str) -> str:
    return prompt


def build_prompt_with_official_context(prompt: str, context_text: str) -> str:
    context_text = (context_text or "").rstrip()
    if not context_text:
        return prompt
    return f"{context_text}\n\n{prompt}"


@torch.no_grad()
def generate_next_line(
    model,
    tokenizer,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int,
    max_input_tokens: int,
) -> str:
    # OOM fallback: if a long prompt causes OOM, halve the token budget and retry
    # (down to a minimum of 512 tokens before giving up).
    budget = max_input_tokens
    while budget >= 512:
        try:
            torch.cuda.empty_cache()
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=budget,
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
            if budget < max_input_tokens:
                print(f"  [oom-fallback] succeeded at budget={budget}")
            return extract_next_line(generated_text)

        except torch.cuda.OutOfMemoryError:
            print(f"  [oom-fallback] OOM at budget={budget}, halving to {budget // 2}")
            torch.cuda.empty_cache()
            budget = budget // 2

    raise RuntimeError(f"OOM even at minimum token budget (512). Giving up on this example.")


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = "left"
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
            attn_implementation="eager",
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        ).to(device)

    model.eval()
    return model, tokenizer, device


def should_use_context(mode: str, context_text: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return bool((context_text or "").strip())


def main() -> None:
    args = parse_args()
    max_rows = maybe_int(args.max_rows)
    output_dir = Path(args.output_dir)
    per_example_csv = output_dir / "crosscodeeval_official_prompts_per_example.csv"
    checkpoint_csv = output_dir / "crosscodeeval_official_prompts_checkpoint.csv"
    summary_txt = output_dir / "crosscodeeval_official_prompts_summary.txt"

    examples = load_crosscodeeval_file(args.input, max_rows=max_rows)
    print(f"Loaded examples: {len(examples)}")
    if not examples:
        raise RuntimeError("No examples loaded.")

    # Resume from checkpoint if present
    prior_rows = load_checkpoint(checkpoint_csv)
    done_ids: set = {r["task_id"] for r in prior_rows}
    if prior_rows:
        print(f"Resuming from checkpoint: {len(prior_rows)} examples already done, skipping.")

    model, tokenizer, device = load_model_and_tokenizer(args.model_name)
    print(f"Loaded model: {args.model_name}")
    print(f"Using device: {device}")

    new_rows: List[Dict[str, Any]] = []
    used_examples = len(prior_rows)
    em_sum = sum(int(r["em"]) for r in prior_rows)
    used_context_count = sum(int(r["used_context"]) for r in prior_rows)
    pending_checkpoint: List[Dict[str, Any]] = []

    for idx, ex in enumerate(examples, start=1):
        task_id = ex["task_id"]
        if task_id in done_ids:
            continue

        gold_line = ex["ground_truth"]
        prompt = ex["prompt"]
        context_text = ex.get("crossfile_context_text", "")
        has_context = should_use_context(args.use_crossfile_context, context_text)
        if has_context:
            used_context_count += 1
            prompt_text = build_prompt_with_official_context(prompt, context_text)
        else:
            prompt_text = build_prompt_only(prompt)

        print(f"[{idx}/{len(examples)}] {task_id}")

        pred = generate_next_line(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            device=device,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
        )
        em = exact_match(pred, gold_line)
        em_sum += em
        used_examples += 1

        context_list = ex.get("crossfile_context_list", []) or []
        top1 = context_list[0] if context_list else {}

        if idx <= args.debug_first_n:
            print("GOLD:", repr(gold_line))
            print("PRED:", repr(pred))
            print("USED_CONTEXT:", has_context)
            if top1:
                print("TOP1_CONTEXT_FILE:", repr(top1.get("filename", "")))
                print("TOP1_CONTEXT_SCORE:", repr(top1.get("score", None)))
            print("-" * 80)

        row = {
            "task_id": ex["task_id"],
            "repo": ex["repo"],
            "path": ex["path"],
            "language": ex["language"],
            "line_no": ex["line_no"],
            "gold_line": gold_line,
            "pred": pred,
            "em": em,
            "used_context": int(has_context),
            "has_crossfile_context_in_file": int(bool(context_text.strip())),
            "num_context_chunks": len(context_list),
            "top1_context_file": top1.get("filename", ""),
            "top1_context_score": top1.get("score", None),
            "prompt_tail": prompt[-500:],
        }
        new_rows.append(row)
        pending_checkpoint.append(row)

        if args.checkpoint_every > 0 and len(pending_checkpoint) >= args.checkpoint_every:
            save_csv(checkpoint_csv, pending_checkpoint, append=True)
            print(f"  [checkpoint] saved {used_examples} examples so far")
            pending_checkpoint = []

    # Flush any remaining rows to checkpoint
    if pending_checkpoint:
        save_csv(checkpoint_csv, pending_checkpoint, append=True)

    if used_examples == 0:
        raise RuntimeError("No usable examples were evaluated.")

    # Write final CSV combining prior + new rows
    all_rows = prior_rows + new_rows
    # Re-cast prior rows' numeric fields (loaded as strings from CSV)
    for r in all_rows:
        r["em"] = int(r["em"])
        r["used_context"] = int(r["used_context"])
        r["has_crossfile_context_in_file"] = int(r["has_crossfile_context_in_file"])
    save_csv(per_example_csv, all_rows)
    em = em_sum / used_examples

    report: List[str] = []
    report.append("CrossCodeEval official prompt-file experiment")
    report.append("")
    report.append(f"Input file: {args.input}")
    report.append(f"Language: {infer_language(args.input)}")
    report.append(f"Model: {args.model_name}")
    report.append(f"Examples requested: {len(examples)}")
    report.append(f"Usable examples: {used_examples}")
    report.append(f"Use cross-file context mode: {args.use_crossfile_context}")
    report.append(f"Examples with context actually used: {used_context_count}")
    report.append("")
    report.append(f"Exact Match (EM): {em:.4f}")
    report.append("")
    report.append(f"Max new tokens: {args.max_new_tokens}")
    report.append(f"Max input tokens: {args.max_input_tokens}")
    report.append(f"Debug first N: {args.debug_first_n}")
    report.append(f"Output dir: {output_dir}")

    save_text(summary_txt, "\n".join(report))
    print(f"Saved per-example CSV to: {per_example_csv}")
    print(f"Saved summary to: {summary_txt}")
    print()
    print("\n".join(report))


if __name__ == "__main__":
    main()
