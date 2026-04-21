
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np


PER_EXAMPLE_CSV = Path(os.getenv("PER_EXAMPLE_CSV", "outputs/completion_step1_1600/repoeval_completion_step1_per_example.csv"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(PER_EXAMPLE_CSV.parent)))
BOOTSTRAP_SAMPLES = int(os.getenv("BOOTSTRAP_SAMPLES", "10000"))
BOOTSTRAP_SEED = int(os.getenv("BOOTSTRAP_SEED", "7"))

TEXT_OUT = OUTPUT_DIR / "repoeval_completion_step1_stats.txt"
JSON_OUT = OUTPUT_DIR / "repoeval_completion_step1_stats.json"


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def to_int_array(rows: List[Dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([int(float(r[key])) for r in rows], dtype=np.int8)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))
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
    logpmfs = []
    log_half = math.log(0.5)
    log_n_fact = math.lgamma(n + 1)
    for k in range(x + 1):
        logpmf = log_n_fact - math.lgamma(k + 1) - math.lgamma(n - k + 1) + n * log_half
        logpmfs.append(logpmf)
    p_one_tail = math.exp(_logsumexp(logpmfs))
    p_two_sided = min(1.0, 2.0 * p_one_tail)
    return a_better, b_better, p_two_sided


def bootstrap_stats(
    prompt_only: np.ndarray,
    bm25: np.ndarray,
    locality: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    n = len(prompt_only)
    rng = np.random.default_rng(seed)

    prompt_rates = np.empty(n_boot, dtype=np.float64)
    bm25_rates = np.empty(n_boot, dtype=np.float64)
    locality_rates = np.empty(n_boot, dtype=np.float64)
    d_bm25_prompt = np.empty(n_boot, dtype=np.float64)
    d_locality_bm25 = np.empty(n_boot, dtype=np.float64)
    d_locality_prompt = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        p = prompt_only[idx].mean()
        b = bm25[idx].mean()
        l = locality[idx].mean()

        prompt_rates[i] = p
        bm25_rates[i] = b
        locality_rates[i] = l
        d_bm25_prompt[i] = b - p
        d_locality_bm25[i] = l - b
        d_locality_prompt[i] = l - p

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
        "delta_bm25_minus_prompt": summarize(d_bm25_prompt),
        "delta_locality_minus_bm25": summarize(d_locality_bm25),
        "delta_locality_minus_prompt": summarize(d_locality_prompt),
    }


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def main() -> None:
    rows = load_rows(PER_EXAMPLE_CSV)
    n = len(rows)

    prompt_only = to_int_array(rows, "em_prompt_only")
    bm25 = to_int_array(rows, "em_bm25")
    locality = to_int_array(rows, "em_locality")

    prompt_s = int(prompt_only.sum())
    bm25_s = int(bm25.sum())
    locality_s = int(locality.sum())

    prompt_rate = prompt_s / n
    bm25_rate = bm25_s / n
    locality_rate = locality_s / n

    prompt_wilson = wilson_ci(prompt_s, n)
    bm25_wilson = wilson_ci(bm25_s, n)
    locality_wilson = wilson_ci(locality_s, n)

    boot = bootstrap_stats(prompt_only, bm25, locality, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)

    bm25_vs_prompt = exact_mcnemar_p(bm25, prompt_only)
    locality_vs_bm25 = exact_mcnemar_p(locality, bm25)
    locality_vs_prompt = exact_mcnemar_p(locality, prompt_only)

    payload = {
        "n_examples": n,
        "metrics": {
            "prompt_only": {
                "successes": prompt_s,
                "em": prompt_rate,
                "wilson_95": {"low": prompt_wilson[0], "high": prompt_wilson[1]},
                "bootstrap_95": {
                    "low": boot["prompt_only_em"]["ci_low"],
                    "high": boot["prompt_only_em"]["ci_high"],
                },
            },
            "bm25": {
                "successes": bm25_s,
                "em": bm25_rate,
                "wilson_95": {"low": bm25_wilson[0], "high": bm25_wilson[1]},
                "bootstrap_95": {
                    "low": boot["bm25_em"]["ci_low"],
                    "high": boot["bm25_em"]["ci_high"],
                },
            },
            "locality": {
                "successes": locality_s,
                "em": locality_rate,
                "wilson_95": {"low": locality_wilson[0], "high": locality_wilson[1]},
                "bootstrap_95": {
                    "low": boot["locality_em"]["ci_low"],
                    "high": boot["locality_em"]["ci_high"],
                },
            },
        },
        "paired_comparisons": {
            "bm25_vs_prompt": {
                "delta": bm25_rate - prompt_rate,
                "bootstrap_95": {
                    "low": boot["delta_bm25_minus_prompt"]["ci_low"],
                    "high": boot["delta_bm25_minus_prompt"]["ci_high"],
                },
                "a_better": bm25_vs_prompt[0],
                "b_better": bm25_vs_prompt[1],
                "mcnemar_exact_p": bm25_vs_prompt[2],
            },
            "locality_vs_bm25": {
                "delta": locality_rate - bm25_rate,
                "bootstrap_95": {
                    "low": boot["delta_locality_minus_bm25"]["ci_low"],
                    "high": boot["delta_locality_minus_bm25"]["ci_high"],
                },
                "a_better": locality_vs_bm25[0],
                "b_better": locality_vs_bm25[1],
                "mcnemar_exact_p": locality_vs_bm25[2],
            },
            "locality_vs_prompt": {
                "delta": locality_rate - prompt_rate,
                "bootstrap_95": {
                    "low": boot["delta_locality_minus_prompt"]["ci_low"],
                    "high": boot["delta_locality_minus_prompt"]["ci_high"],
                },
                "a_better": locality_vs_prompt[0],
                "b_better": locality_vs_prompt[1],
                "mcnemar_exact_p": locality_vs_prompt[2],
            },
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = []
    lines.append("RepoEval completion step-1 statistics")
    lines.append("")
    lines.append(f"Per-example CSV: {PER_EXAMPLE_CSV}")
    lines.append(f"Examples: {n}")
    lines.append(f"Bootstrap samples: {BOOTSTRAP_SAMPLES}")
    lines.append("")
    lines.append(
        f"Prompt-only EM: {pct(prompt_rate)} ({prompt_s}/{n}), "
        f"Wilson 95% CI [{pct(prompt_wilson[0])}, {pct(prompt_wilson[1])}], "
        f"bootstrap 95% CI [{pct(boot['prompt_only_em']['ci_low'])}, {pct(boot['prompt_only_em']['ci_high'])}]"
    )
    lines.append(
        f"BM25 EM: {pct(bm25_rate)} ({bm25_s}/{n}), "
        f"Wilson 95% CI [{pct(bm25_wilson[0])}, {pct(bm25_wilson[1])}], "
        f"bootstrap 95% CI [{pct(boot['bm25_em']['ci_low'])}, {pct(boot['bm25_em']['ci_high'])}]"
    )
    lines.append(
        f"Locality EM: {pct(locality_rate)} ({locality_s}/{n}), "
        f"Wilson 95% CI [{pct(locality_wilson[0])}, {pct(locality_wilson[1])}], "
        f"bootstrap 95% CI [{pct(boot['locality_em']['ci_low'])}, {pct(boot['locality_em']['ci_high'])}]"
    )
    lines.append("")
    lines.append(
        f"BM25 - Prompt-only: Δ={pct(bm25_rate - prompt_rate)}, "
        f"bootstrap 95% CI [{pct(boot['delta_bm25_minus_prompt']['ci_low'])}, {pct(boot['delta_bm25_minus_prompt']['ci_high'])}], "
        f"McNemar exact p={bm25_vs_prompt[2]:.6g}, wins/losses={bm25_vs_prompt[0]}/{bm25_vs_prompt[1]}"
    )
    lines.append(
        f"Locality - BM25: Δ={pct(locality_rate - bm25_rate)}, "
        f"bootstrap 95% CI [{pct(boot['delta_locality_minus_bm25']['ci_low'])}, {pct(boot['delta_locality_minus_bm25']['ci_high'])}], "
        f"McNemar exact p={locality_vs_bm25[2]:.6g}, wins/losses={locality_vs_bm25[0]}/{locality_vs_bm25[1]}"
    )
    lines.append(
        f"Locality - Prompt-only: Δ={pct(locality_rate - prompt_rate)}, "
        f"bootstrap 95% CI [{pct(boot['delta_locality_minus_prompt']['ci_low'])}, {pct(boot['delta_locality_minus_prompt']['ci_high'])}], "
        f"McNemar exact p={locality_vs_prompt[2]:.6g}, wins/losses={locality_vs_prompt[0]}/{locality_vs_prompt[1]}"
    )

    TEXT_OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print()
    print(f"Saved JSON to: {JSON_OUT}")
    print(f"Saved text summary to: {TEXT_OUT}")


if __name__ == "__main__":
    main()
