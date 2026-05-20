"""
Compute statistics from experimental outputs:
  - McNemar exact test (p-value + chi2 statistic) for all pairwise comparisons
  - Odds ratio (OR = n_B_wins / n_A_wins) as effect size
  - Cliff's delta for binary outcomes
  - Bootstrap 95% CI (seed=7, 10000 resamples) for all EM differences
  - Dense EM statistical test (previously missing entirely)

Output: outputs/paper_stats/paper_statistics.json + paper_statistics.txt
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "outputs" / "paper_stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_SEED = 7
BOOTSTRAP_N = 10_000


# ── Statistical helpers ───────────────────────────────────────────────────────

def exact_mcnemar(a: np.ndarray, b: np.ndarray):
    """
    Compare binary arrays A and B (same N examples).
    Convention: we test B > A, i.e. 'does B improve over A?'

    Returns dict with:
      n_a_only  : A correct, B wrong  (A wins)
      n_b_only  : B correct, A wrong  (B wins)
      chi2      : asymptotic chi-squared(1)
      p_exact   : exact binomial p-value (two-sided)
      OR        : odds ratio = n_b_only / n_a_only  (B over A)
      cliff_delta: (n_b_only - n_a_only) / N
    """
    assert len(a) == len(b), "Arrays must be same length"
    n = len(a)
    n_a_only = int(np.sum((a == 1) & (b == 0)))   # A wins, B loses
    n_b_only = int(np.sum((a == 0) & (b == 1)))   # B wins, A loses
    n_disc = n_a_only + n_b_only

    # Exact two-sided p (binomial, H0: p_B_wins = 0.5)
    if n_disc == 0:
        p_exact = 1.0
        chi2 = 0.0
    else:
        p_exact = float(2.0 * stats.binom.cdf(
            min(n_a_only, n_b_only), n_disc, 0.5))
        p_exact = min(p_exact, 1.0)
        chi2 = float((n_b_only - n_a_only) ** 2 / n_disc)

    OR = (n_b_only / n_a_only) if n_a_only > 0 else float("inf")
    cliff_delta = float((n_b_only - n_a_only) / n)

    return dict(
        n=n,
        n_a_only=n_a_only,
        n_b_only=n_b_only,
        chi2=round(chi2, 4),
        p_exact=p_exact,
        OR=round(OR, 4),
        cliff_delta=round(cliff_delta, 4),
    )


def bootstrap_ci(a: np.ndarray, b: np.ndarray,
                 n_resamples: int = BOOTSTRAP_N,
                 seed: int = BOOTSTRAP_SEED):
    """Bootstrap 95% CI for EM(B) - EM(A) in percentage points."""
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = b[idx].mean() - a[idx].mean()
    lo = float(np.percentile(diffs, 2.5)) * 100
    hi = float(np.percentile(diffs, 97.5)) * 100
    return round(lo, 2), round(hi, 2)


def analyse_pair(label_a, label_b, a: np.ndarray, b: np.ndarray):
    """Full analysis for a single A-vs-B pairwise comparison."""
    mc = exact_mcnemar(a, b)
    ci_lo, ci_hi = bootstrap_ci(a, b)
    em_a = round(float(a.mean()) * 100, 2)
    em_b = round(float(b.mean()) * 100, 2)
    delta_pp = round(em_b - em_a, 2)
    return dict(
        comparison=f"{label_b} vs {label_a}",
        em_a=em_a,
        em_b=em_b,
        delta_pp=delta_pp,
        ci_95_pp=[ci_lo, ci_hi],
        n=mc["n"],
        n_a_wins=mc["n_a_only"],
        n_b_wins=mc["n_b_only"],
        chi2_1=mc["chi2"],
        p_exact=mc["p_exact"],
        OR=mc["OR"],
        cliff_delta=mc["cliff_delta"],
    )


# ── Data paths ────────────────────────────────────────────────────────────────

MODELS = [
    ("StarCoder2-7B",
     ROOT / "outputs/completion_step1_1600/repoeval_completion_step1_per_example.csv",
     "em_prompt_only", "em_bm25", "em_locality"),
    ("StarCoder2-3B",
     ROOT / "outputs/completion_starcoder2_3b_1600/repoeval_completion_step1_per_example.csv",
     "em_prompt_only", "em_bm25", "em_locality"),
    ("CodeLlama-7B",
     ROOT / "outputs/completion_codellama_7b_1600/completion_codellama_7b_1600_per_example.csv",
     "em_prompt_only", "em_bm25", "em_locality"),
    ("DeepSeek-Coder-6.7B",
     ROOT / "outputs/completion_deepseek_6b_1600/completion_deepseek_6b_1600_per_example.csv",
     "em_prompt_only", "em_bm25", "em_locality"),
    ("CodeQwen1.5-7B",
     ROOT / "outputs/completion_codeqwen_7b_1600/completion_codeqwen_7b_1600_per_example.csv",
     "em_prompt_only", "em_bm25", "em_locality"),
]

DENSE_CSV = ROOT / "outputs/completion_dense_starcoder2_7b/dense_completion_per_example.csv"


# ── Run all comparisons ───────────────────────────────────────────────────────

results = {}

print("=" * 70)
print("RQ4: End-to-end EM — 5 models (BM25 prompt-only vs BM25 vs Locality)")
print("=" * 70)

rq4 = {}
for model_name, csv_path, col_po, col_bm25, col_loc in MODELS:
    df = pd.read_csv(csv_path)
    po  = df[col_po].values.astype(int)
    bm  = df[col_bm25].values.astype(int)
    loc = df[col_loc].values.astype(int)

    bm_vs_po  = analyse_pair("Prompt-only", "BM25",     po,  bm)
    loc_vs_bm = analyse_pair("BM25",        "Locality", bm,  loc)

    rq4[model_name] = {
        "BM25_vs_PromptOnly": bm_vs_po,
        "Locality_vs_BM25":   loc_vs_bm,
    }

    print(f"\n  {model_name}")
    for label, r in [("BM25 vs Prompt-only", bm_vs_po),
                     ("Locality vs BM25",    loc_vs_bm)]:
        print(f"    {label}:")
        print(f"      EM: {r['em_a']}% → {r['em_b']}%  Δ={r['delta_pp']:+.2f} pp")
        print(f"      95% CI: [{r['ci_95_pp'][0]}, {r['ci_95_pp'][1]}] pp")
        print(f"      McNemar: χ²(1)={r['chi2_1']:.3f},  p={r['p_exact']:.3e}")
        print(f"      OR={r['OR']:.3f},  Cliff's δ={r['cliff_delta']:.4f}")
        print(f"      discordant: {r['n_b_wins']} B-wins, {r['n_a_wins']} A-wins")

results["RQ4"] = rq4

print("\n" + "=" * 70)
print("RQ5: Dense EM — Dense vs Dense+Locality (StarCoder2-7B, raw format)")
print("=" * 70)

df_dense = pd.read_csv(DENSE_CSV)
dense     = df_dense["em_dense"].values.astype(int)
dense_loc = df_dense["em_dense_loc"].values.astype(int)

dense_result = analyse_pair("Dense", "Dense+Locality", dense, dense_loc)
results["RQ5_dense_EM"] = dense_result

print(f"\n  Dense vs Dense+Locality:")
print(f"    EM: {dense_result['em_a']}% → {dense_result['em_b']}%"
      f"  Δ={dense_result['delta_pp']:+.2f} pp")
print(f"    95% CI: [{dense_result['ci_95_pp'][0]}, {dense_result['ci_95_pp'][1]}] pp")
print(f"    McNemar: χ²(1)={dense_result['chi2_1']:.3f},  p={dense_result['p_exact']:.3e}")
print(f"    OR={dense_result['OR']:.3f},  Cliff's δ={dense_result['cliff_delta']:.4f}")
print(f"    discordant: {dense_result['n_b_wins']} B-wins,"
      f" {dense_result['n_a_wins']} A-wins")


# ── Save ──────────────────────────────────────────────────────────────────────

json_out = OUT_DIR / "paper_statistics.json"
with open(json_out, "w") as f:
    json.dump(results, f, indent=2)

# Human-readable summary
lines = []
lines.append("PAPER STATISTICS\n")
lines.append("=" * 70)
lines.append("\nRQ4 — End-to-End EM (5 models)\n")

for model_name, comps in results["RQ4"].items():
    lines.append(f"\n  {model_name}")
    for comp_label, r in comps.items():
        lines.append(f"    {r['comparison']}:")
        lines.append(f"      EM {r['em_a']}% → {r['em_b']}%  Δ={r['delta_pp']:+.2f} pp")
        lines.append(f"      Bootstrap 95% CI [{r['ci_95_pp'][0]}, {r['ci_95_pp'][1]}] pp")
        lines.append(f"      McNemar χ²(1)={r['chi2_1']:.3f}, p={r['p_exact']:.3e}")
        lines.append(f"      OR={r['OR']:.3f}  (n_wins={r['n_b_wins']}, n_losses={r['n_a_wins']})")
        lines.append(f"      Cliff's delta={r['cliff_delta']:.4f}")

lines.append("\n\nRQ5 — Dense EM (StarCoder2-7B, raw prompt)\n")
r = results["RQ5_dense_EM"]
lines.append(f"  {r['comparison']}:")
lines.append(f"    EM {r['em_a']}% → {r['em_b']}%  Δ={r['delta_pp']:+.2f} pp")
lines.append(f"    Bootstrap 95% CI [{r['ci_95_pp'][0]}, {r['ci_95_pp'][1]}] pp")
lines.append(f"    McNemar χ²(1)={r['chi2_1']:.3f}, p={r['p_exact']:.3e}")
lines.append(f"    OR={r['OR']:.3f}  (n_wins={r['n_b_wins']}, n_losses={r['n_a_wins']})")
lines.append(f"    Cliff's delta={r['cliff_delta']:.4f}")

txt_out = OUT_DIR / "paper_statistics.txt"
with open(txt_out, "w") as f:
    f.write("\n".join(lines))

print(f"\nSaved:\n  {json_out}\n  {txt_out}")
