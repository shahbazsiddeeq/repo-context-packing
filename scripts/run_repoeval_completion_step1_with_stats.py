
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
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

if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

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

BOOTSTRAP_SAMPLES = int(os.getenv("BOOTSTRAP_SAMPLES", "10000"))
BOOTSTRAP_SEED = int(os.getenv("BOOTSTRAP_SEED", "7"))

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/completion_step1_smoketest"))
PER_EXAMPLE_CSV = OUTPUT_DIR / "repoeval_completion_step1_per_example.csv"
SUMMARY_TXT = OUTPUT_DIR / "repoeval_completion_step1_summary.txt"
STATS_TXT = OUTPUT_DIR / "repoeval_completion_step1_stats.txt"
STATS_JSON = OUTPUT_DIR / "repoeval_completion_step1_stats.json"


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


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    if not text:
        return ""
    line = text.split("\n", 1)[0]
    return line.rstrip()


def build_prompt_only(target_path: str, prompt: str) -> str:
    return prompt


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
    return extract_next_line(generated_text)


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

    common_kwargs = dict(
        attn_implementation="eager",
    )

    if device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            **common_kwargs,
        )
    elif device.type == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            **common_kwargs,
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            **common_kwargs,
        ).to(device)

    model.eval()
    return model, tokenizer, device


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4 * n * n))
    return max(0.0, center - margin), min(1.0, center + margin)


def _logsumexp(vals: List[float]) -> float:
    if not vals:
        return float("-inf")
    m = max(vals)
    if math.isinf(m):
        return m
    return m + math.log(sum(math.exp(v - m) for v in vals))


def exact_mcnemar_p(a: np.ndarray, b: np.ndarray) -> Tuple[int, int, float]:
    a_better = int(np.sum((a == 1) & (b == 0)))
    b_better = int(np.sum((a == 0) & (b == 1)))
    n = a_better + b_better
    if n == 0:
        return a_better, b_better, 1.0

    x = min(a_better, b_better)
    log_n_fact = math.lgamma(n + 1)
    log_half = math.log(0.5)
    logpmfs = [
        log_n_fact - math.lgamma(k + 1) - math.lgamma(n - k + 1) + n * log_half
        for k in range(x + 1)
    ]
    p_one_tail = math.exp(_logsumexp(logpmfs))
    p_two_sided = min(1.0, 2.0 * p_one_tail)
    return a_better, b_better, p_two_sided


def bootstrap_stats(prompt_only: np.ndarray, bm25: np.ndarray, locality: np.ndarray) -> Dict[str, Dict[str, float]]:
    n = len(prompt_only)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    prompt_rates = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    bm25_rates = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    locality_rates = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    d_bp = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    d_lb = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    d_lp = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)

    for i in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, n, size=n)
        p = prompt_only[idx].mean()
        b = bm25[idx].mean()
        l = locality[idx].mean()
        prompt_rates[i] = p
        bm25_rates[i] = b
        locality_rates[i] = l
        d_bp[i] = b - p
        d_lb[i] = l - b
        d_lp[i] = l - p

    def summarize(arr: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(arr.mean()),
            "ci_low": float(np.quantile(arr, 0.025)),
            "ci_high": float(np.quantile(arr, 0.975)),
        }

    return {
        "prompt_only_em": summarize(prompt_rates),
        "bm25_em": summarize(bm25_rates),
        "locality_em": summarize(locality_rates),
        "delta_bm25_minus_prompt": summarize(d_bp),
        "delta_locality_minus_bm25": summarize(d_lb),
        "delta_locality_minus_prompt": summarize(d_lp),
    }


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


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
        retrieval_query = ex.get("query", "")
        generation_prefix = ex.get("prompt", retrieval_query)
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

    prompt_arr = np.asarray([r["em_prompt_only"] for r in rows], dtype=np.int8)
    bm25_arr = np.asarray([r["em_bm25"] for r in rows], dtype=np.int8)
    locality_arr = np.asarray([r["em_locality"] for r in rows], dtype=np.int8)

    em_prompt_only = em_prompt_only_sum / used_examples
    em_bm25 = em_bm25_sum / used_examples
    em_locality = em_locality_sum / used_examples

    prompt_wilson = wilson_ci(em_prompt_only_sum, used_examples)
    bm25_wilson = wilson_ci(em_bm25_sum, used_examples)
    locality_wilson = wilson_ci(em_locality_sum, used_examples)

    boot = bootstrap_stats(prompt_arr, bm25_arr, locality_arr)

    bm25_vs_prompt = exact_mcnemar_p(bm25_arr, prompt_arr)
    locality_vs_bm25 = exact_mcnemar_p(locality_arr, bm25_arr)
    locality_vs_prompt = exact_mcnemar_p(locality_arr, prompt_arr)

    stats_payload = {
        "n_examples": used_examples,
        "metrics": {
            "prompt_only": {
                "successes": int(em_prompt_only_sum),
                "em": em_prompt_only,
                "wilson_95": {"low": prompt_wilson[0], "high": prompt_wilson[1]},
                "bootstrap_95": {"low": boot["prompt_only_em"]["ci_low"], "high": boot["prompt_only_em"]["ci_high"]},
            },
            "bm25": {
                "successes": int(em_bm25_sum),
                "em": em_bm25,
                "wilson_95": {"low": bm25_wilson[0], "high": bm25_wilson[1]},
                "bootstrap_95": {"low": boot["bm25_em"]["ci_low"], "high": boot["bm25_em"]["ci_high"]},
            },
            "locality": {
                "successes": int(em_locality_sum),
                "em": em_locality,
                "wilson_95": {"low": locality_wilson[0], "high": locality_wilson[1]},
                "bootstrap_95": {"low": boot["locality_em"]["ci_low"], "high": boot["locality_em"]["ci_high"]},
            },
        },
        "paired_comparisons": {
            "bm25_vs_prompt": {
                "delta": em_bm25 - em_prompt_only,
                "bootstrap_95": {"low": boot["delta_bm25_minus_prompt"]["ci_low"], "high": boot["delta_bm25_minus_prompt"]["ci_high"]},
                "wins": bm25_vs_prompt[0],
                "losses": bm25_vs_prompt[1],
                "mcnemar_exact_p": bm25_vs_prompt[2],
            },
            "locality_vs_bm25": {
                "delta": em_locality - em_bm25,
                "bootstrap_95": {"low": boot["delta_locality_minus_bm25"]["ci_low"], "high": boot["delta_locality_minus_bm25"]["ci_high"]},
                "wins": locality_vs_bm25[0],
                "losses": locality_vs_bm25[1],
                "mcnemar_exact_p": locality_vs_bm25[2],
            },
            "locality_vs_prompt": {
                "delta": em_locality - em_prompt_only,
                "bootstrap_95": {"low": boot["delta_locality_minus_prompt"]["ci_low"], "high": boot["delta_locality_minus_prompt"]["ci_high"]},
                "wins": locality_vs_prompt[0],
                "losses": locality_vs_prompt[1],
                "mcnemar_exact_p": locality_vs_prompt[2],
            },
        },
    }
    save_json(STATS_JSON, stats_payload)

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
    report.append(f"Prompt-only EM: {em_prompt_only:.4f} ({pct(em_prompt_only)}), Wilson 95% CI [{pct(prompt_wilson[0])}, {pct(prompt_wilson[1])}]")
    report.append(f"BM25 top-1 context EM: {em_bm25:.4f} ({pct(em_bm25)}), Wilson 95% CI [{pct(bm25_wilson[0])}, {pct(bm25_wilson[1])}]")
    report.append(f"Locality top-1 context EM: {em_locality:.4f} ({pct(em_locality)}), Wilson 95% CI [{pct(locality_wilson[0])}, {pct(locality_wilson[1])}]")
    report.append("")
    report.append(
        f"Delta (BM25 - Prompt-only): {em_bm25 - em_prompt_only:.4f} ({pct(em_bm25 - em_prompt_only)}), "
        f"bootstrap 95% CI [{pct(boot['delta_bm25_minus_prompt']['ci_low'])}, {pct(boot['delta_bm25_minus_prompt']['ci_high'])}], "
        f"McNemar exact p={bm25_vs_prompt[2]:.6g}"
    )
    report.append(
        f"Delta (Locality - BM25): {em_locality - em_bm25:.4f} ({pct(em_locality - em_bm25)}), "
        f"bootstrap 95% CI [{pct(boot['delta_locality_minus_bm25']['ci_low'])}, {pct(boot['delta_locality_minus_bm25']['ci_high'])}], "
        f"McNemar exact p={locality_vs_bm25[2]:.6g}"
    )
    report.append(
        f"Delta (Locality - Prompt-only): {em_locality - em_prompt_only:.4f} ({pct(em_locality - em_prompt_only)}), "
        f"bootstrap 95% CI [{pct(boot['delta_locality_minus_prompt']['ci_low'])}, {pct(boot['delta_locality_minus_prompt']['ci_high'])}], "
        f"McNemar exact p={locality_vs_prompt[2]:.6g}"
    )
    report.append("")
    report.append(f"Chunk size: {CHUNK_SIZE}")
    report.append(f"Chunk overlap: {CHUNK_OVERLAP}")
    report.append(f"Top-k retrieval: {TOP_K}")
    report.append(f"Same-file bonus: {SAME_FILE_BONUS}")
    report.append(f"Line-distance weight: {LINE_DISTANCE_WEIGHT}")
    report.append(f"Max new tokens: {MAX_NEW_TOKENS}")
    report.append(f"Max input tokens: {MAX_INPUT_TOKENS}")
    report.append(f"Bootstrap samples: {BOOTSTRAP_SAMPLES}")
    report.append(f"Debug first N: {DEBUG_FIRST_N}")
    report.append(f"Output dir: {OUTPUT_DIR}")

    stats_lines = []
    stats_lines.append("RepoEval completion step-1 paired statistics")
    stats_lines.append("")
    stats_lines.append(f"BM25 vs Prompt-only wins/losses: {bm25_vs_prompt[0]}/{bm25_vs_prompt[1]}")
    stats_lines.append(f"Locality vs BM25 wins/losses: {locality_vs_bm25[0]}/{locality_vs_bm25[1]}")
    stats_lines.append(f"Locality vs Prompt-only wins/losses: {locality_vs_prompt[0]}/{locality_vs_prompt[1]}")

    save_text(SUMMARY_TXT, "\n".join(report))
    save_text(STATS_TXT, "\n".join(stats_lines))

    print(f"Saved per-example CSV to: {PER_EXAMPLE_CSV}")
    print(f"Saved summary to: {SUMMARY_TXT}")
    print(f"Saved stats text to: {STATS_TXT}")
    print(f"Saved stats JSON to: {STATS_JSON}")
    print()
    print("\n".join(report))
    print()
    print("\n".join(stats_lines))


if __name__ == "__main__":
    main()
