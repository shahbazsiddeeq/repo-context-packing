from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:  # pragma: no cover
    AutoTokenizer = None
    AutoModelForCausalLM = None

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
# Main experiment settings
# -----------------------------
# Use GENERATION_BACKEND=api to call a remote API.
# Use GENERATION_BACKEND=local to load the model with Hugging Face.
GENERATION_BACKEND = os.getenv("GENERATION_BACKEND", "api").lower()

# Local backend settings
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "bigcode/starcoder2-7b"")

# API backend settings
# Provide the FULL endpoint URL for your serving stack.
# Examples:
#   OpenAI-compatible completions: http://host:8000/v1/completions
#   OpenAI-compatible chat:        http://host:8000/v1/chat/completions
#   TGI generate endpoint:         http://host:8080/generate
#   Ollama-style generate endpoint:  http://host:11434/api/generate
API_MODEL_NAME = os.getenv("API_MODEL_NAME", "starcoder2:7b")
API_URL = os.getenv("STARCODER_API_URL", "http://localhost:11434/api/generate")
API_KEY = os.getenv("STARCODER_API_KEY", "")
API_FORMAT = os.getenv("STARCODER_API_FORMAT", "openai_completions").lower()
API_TIMEOUT_SEC = int(os.getenv("STARCODER_API_TIMEOUT_SEC", "120"))
API_MAX_RETRIES = int(os.getenv("STARCODER_API_MAX_RETRIES", "3"))
API_NUM_CTX = int(os.getenv("STARCODER_API_NUM_CTX", "4096"))
API_TEMPERATURE = float(os.getenv("STARCODER_API_TEMPERATURE", "0.0"))
API_TOP_P = float(os.getenv("STARCODER_API_TOP_P", "1.0"))

MAX_ROWS = 1600
TOP_K = 10
MAX_FILES_PER_REPO = 1000

CHUNK_SIZE = 120
CHUNK_OVERLAP = 20

SAME_FILE_BONUS = 150.0
LINE_DISTANCE_WEIGHT = 0.2

MAX_NEW_TOKENS = 32

# OUTPUT_DIR = Path("outputs/completion_step1")
OUTPUT_DIR = Path("outputs/completion_step1_smoketest")
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


def extract_next_line(generated_text: str) -> str:
    text = generated_text.replace("\r\n", "\n").replace("\r", "\n")
    line = text.split("\n", 1)[0]
    return line.rstrip()


def build_prompt_only(target_path: str, prompt: str) -> str:
    return (
        "### Current File Path\n"
        f"{target_path}\n\n"
        "### Current Code Prefix\n"
        f"{prompt}\n\n"
        "### Task\n"
        "Predict the next line of code only.\n"
    )


def build_prompt_with_context(target_path: str, prompt: str, context_text: str) -> str:
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


def choose_device():
    if torch is None:
        raise RuntimeError("torch is not installed, so the local backend cannot be used.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_local_model_and_tokenizer(model_name: str):
    if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
        raise RuntimeError(
            "transformers/torch are not installed, so the local backend cannot be used."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = choose_device()

    if device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
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


@torch.no_grad() if torch is not None else (lambda f: f)
def generate_next_line_local(
    model,
    tokenizer,
    prompt_text: str,
    device,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=16000)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return extract_next_line(generated_text)


def _build_api_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


def _call_api_once(prompt_text: str, max_new_tokens: int) -> str:
    if not API_URL:
        raise RuntimeError(
            "STARCODER_API_URL is empty. Set it to the full inference endpoint URL."
        )

    headers = _build_api_headers()

    if API_FORMAT == "openai_completions":
        payload = {
            "model": API_MODEL_NAME,
            "prompt": prompt_text,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
        }
    elif API_FORMAT == "openai_chat":
        payload = {
            "model": API_MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a code completion model. Return only the next line of code.",
                },
                {"role": "user", "content": prompt_text},
            ],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
        }
    elif API_FORMAT == "tgi":
        payload = {
            "inputs": prompt_text,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "return_full_text": False,
                "temperature": API_TEMPERATURE,
                "top_p": API_TOP_P,
            },
        }
    elif API_FORMAT == "ollama_generate":
        payload = {
            "model": API_MODEL_NAME,
            "prompt": prompt_text,
            "stream": False,
            "options": {
                "num_ctx": API_NUM_CTX,
                "temperature": API_TEMPERATURE,
                "top_p": API_TOP_P,
                "num_predict": max_new_tokens,
            },
        }
    else:
        raise RuntimeError(
            f"Unsupported STARCODER_API_FORMAT={API_FORMAT!r}. "
            "Use one of: openai_completions, openai_chat, tgi, ollama_generate"
        )

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=API_TIMEOUT_SEC,
    )
    response.raise_for_status()
    data = response.json()

    if API_FORMAT == "openai_completions":
        return data["choices"][0]["text"]
    if API_FORMAT == "openai_chat":
        return data["choices"][0]["message"]["content"]
    if API_FORMAT == "ollama_generate":
        if isinstance(data, dict) and "response" in data:
            return data["response"]
        raise RuntimeError(f"Unexpected ollama_generate response: {data}")

    # TGI can return either a dict or a list depending on the deployment stack.
    if isinstance(data, list):
        return data[0]["generated_text"]
    return data["generated_text"]


def generate_next_line_api(prompt_text: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            generated_text = _call_api_once(prompt_text, max_new_tokens=max_new_tokens)
            return extract_next_line(generated_text)
        except Exception as exc:
            last_error = exc
            if attempt == API_MAX_RETRIES:
                break
            time.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError(f"API generation failed after {API_MAX_RETRIES} attempts: {last_error}")


def generate_next_line(
    backend: str,
    prompt_text: str,
    model=None,
    tokenizer=None,
    device=None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    if backend == "api":
        return generate_next_line_api(prompt_text, max_new_tokens=max_new_tokens)
    if backend == "local":
        return generate_next_line_local(
            model,
            tokenizer,
            prompt_text,
            device,
            max_new_tokens=max_new_tokens,
        )
    raise RuntimeError(f"Unsupported backend: {backend!r}")


def main():
    examples = load_repoeval_file(REPOEVAL_FILE, max_rows=MAX_ROWS)
    print(f"Loaded examples: {len(examples)}")

    model = None
    tokenizer = None
    device = None

    if GENERATION_BACKEND == "local":
        model, tokenizer, device = load_local_model_and_tokenizer(LOCAL_MODEL_NAME)
        print(f"Loaded local model: {LOCAL_MODEL_NAME}")
        print(f"Using device: {device}")
    elif GENERATION_BACKEND == "api":
        print(f"Using API backend: {API_FORMAT}")
        print(f"API model name: {API_MODEL_NAME}")
        print(f"API URL: {API_URL}")
    else:
        raise RuntimeError(
            f"Unsupported GENERATION_BACKEND={GENERATION_BACKEND!r}. Use 'api' or 'local'."
        )

    rows: List[Dict[str, Any]] = []

    used_examples = 0
    skipped_repo = 0
    skipped_files = 0
    skipped_chunks = 0
    skipped_retrieval = 0
    skipped_generation = 0

    em_prompt_only_sum = 0
    em_bm25_sum = 0
    em_locality_sum = 0

    for idx, ex in enumerate(examples, start=1):
        repo_name = ex["repo"]
        target_path = ex["path"]

        # Use the retrieval query for retrieval, but the full prefix for generation.
        retrieval_query = ex["query"]
        generation_prompt = ex.get("prompt", ex["query"])

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

        bm25_top1 = bm25_results[0]
        locality_top1 = locality_results[0]

        prompt_only_text = build_prompt_only(target_path, generation_prompt)
        bm25_prompt_text = build_prompt_with_context(target_path, generation_prompt, bm25_top1.text)
        locality_prompt_text = build_prompt_with_context(target_path, generation_prompt, locality_top1.text)

        try:
            pred_prompt_only = generate_next_line(
                GENERATION_BACKEND,
                prompt_only_text,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
            pred_bm25 = generate_next_line(
                GENERATION_BACKEND,
                bm25_prompt_text,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
            pred_locality = generate_next_line(
                GENERATION_BACKEND,
                locality_prompt_text,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
        except Exception as exc:
            skipped_generation += 1
            print(f"  Generation failed: {exc}")
            continue

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
                "generation_backend": GENERATION_BACKEND,
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
    report.append(f"Generation backend: {GENERATION_BACKEND}")
    if GENERATION_BACKEND == "local":
        report.append(f"Local model: {LOCAL_MODEL_NAME}")
    else:
        report.append(f"API model: {API_MODEL_NAME}")
        report.append(f"API format: {API_FORMAT}")
        report.append(f"API URL: {API_URL}")
    report.append(f"Examples requested: {len(examples)}")
    report.append(f"Usable examples: {used_examples}")
    report.append(f"Skipped repo: {skipped_repo}")
    report.append(f"Skipped files: {skipped_files}")
    report.append(f"Skipped chunks: {skipped_chunks}")
    report.append(f"Skipped retrieval: {skipped_retrieval}")
    report.append(f"Skipped generation: {skipped_generation}")
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

    save_text(SUMMARY_TXT, "\n".join(report))

    print(f"Saved per-example CSV to: {PER_EXAMPLE_CSV}")
    print(f"Saved summary to: {SUMMARY_TXT}")
    print()
    print("\n".join(report))


if __name__ == "__main__":
    main()
