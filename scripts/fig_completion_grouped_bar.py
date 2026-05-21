"""
fig_completion_grouped_bar.py
------------------------------
Generates the grouped bar chart for RQ4: end-to-end exact-match (EM%)
across 5 model architectures and 3 retrieval conditions.

Data sourced from table_completion.tex (verified against per-example CSVs).
Includes a zoomed inset panel showing the BM25 vs. Locality comparison only.

Outputs: PDF (vector) and PNG (300 dpi) in paper/figures/.

Usage:
    python scripts/fig_completion_grouped_bar.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_DIR  = os.path.join(os.path.dirname(REPO_ROOT), "paper")
OUT_DIR    = os.path.join(PAPER_DIR, "figures")
OUT_PDF   = os.path.join(OUT_DIR, "fig_completion_grouped_bar.pdf")
OUT_PNG   = os.path.join(OUT_DIR, "fig_completion_grouped_bar.png")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data from table_completion.tex ────────────────────────────────────────────
# Model short labels (x-axis)
models = [
    "StarCoder2\n7B",
    "StarCoder2\n3B",
    "CodeLlama\n7B",
    "DeepSeek-Coder\n6.7B",
    "CodeQwen1.5\n7B",
]

# EM percentages for each condition
em_prompt = np.array([34.62, 33.75, 33.25, 35.56, 40.31])
em_bm25   = np.array([78.19, 77.62, 76.06, 76.62, 78.12])
em_local  = np.array([79.38, 79.19, 77.56, 78.38, 79.50])

# Bootstrap 95% CI half-widths for Locality top-1 (from table)
# CI format [lo, hi] deltas vs BM25 -> half-width for error bar on absolute EM
# [0.44, 2.00], [0.75, 2.44], [0.69, 2.31], [0.94, 2.62], [0.56, 2.25]
ci_lo = np.array([0.44, 0.75, 0.69, 0.94, 0.56])
ci_hi = np.array([2.00, 2.44, 2.31, 2.62, 2.25])
# Convert to absolute EM CIs
local_ci_lo = em_bm25 + ci_lo   # lower bound absolute
local_ci_hi = em_bm25 + ci_hi   # upper bound absolute
# Asymmetric error bars in EM%:
err_lo = em_local - local_ci_lo  # down from locality EM
err_hi = local_ci_hi - em_local  # up from locality EM

# ── Plot setup ────────────────────────────────────────────────────────────────
n_models = len(models)
x        = np.arange(n_models)
width    = 0.24

# Colors: prompt-only (gray), BM25 (light steel blue), Locality (light green)
C_PROMPT = "#adb5bd"   # neutral gray
C_BM25   = "#4393c3"   # medium blue
C_LOCAL  = "#74c476"   # medium green (our method)

# ── Two-panel figure: left = full range, right = zoomed BM25 vs. Locality ────
fig, (ax, ax_in) = plt.subplots(
    1, 2, figsize=(12, 5.2),
    gridspec_kw={"width_ratios": [2.8, 1], "wspace": 0.28}
)

# ── LEFT PANEL: full range bar chart ─────────────────────────────────────────
b1 = ax.bar(x - width,     em_prompt, width, label="Prompt-only (no retrieval)",
            color=C_PROMPT, edgecolor="white", linewidth=0.5, zorder=3)
b2 = ax.bar(x,             em_bm25,   width, label="BM25 top-1",
            color=C_BM25,  edgecolor="white", linewidth=0.5, zorder=3)
b3 = ax.bar(x + width,     em_local,  width, label="Locality top-1 (ours)",
            color=C_LOCAL, edgecolor="white", linewidth=0.5, zorder=3,
            yerr=[err_lo, err_hi], error_kw=dict(
                ecolor="#555", capsize=3.5, capthick=1.2, elinewidth=1.2))

# Deltas hardcoded from exact counts (not rounded EM percentages) to match paper text.
# Computing from rounded values introduces rounding artefacts: e.g. 79.19-77.62=1.57
# but actual gain (1267-1242)/1600*100 = 1.5625 → 1.56 pp as reported in the paper.
deltas = np.array([1.19, 1.56, 1.50, 1.75, 1.38])
for i, (bar, d) in enumerate(zip(b3, deltas)):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + err_hi[i] + 0.6,
            f"+{d:.2f}pp", ha="center", va="bottom",
            fontsize=8, color="#2d6a2d", fontweight="bold")

# Prompt-only value labels
for bar, v in zip(b1, em_prompt):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"{v:.1f}%", ha="center", va="bottom",
            fontsize=7, color="#555")

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel("Exact Match (%)", fontsize=11)
ax.set_ylim(0, 95)
ax.set_xlim(-0.55, n_models - 0.45)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("(a) Full range: Prompt-only vs. BM25 vs. Locality", fontsize=10, pad=8)
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9, ncol=1)

# ── RIGHT PANEL: zoomed BM25 vs. Locality only ───────────────────────────────
inset_width = 0.32
for i in range(n_models):
    ax_in.bar(i - inset_width/2, em_bm25[i],  inset_width,
              color=C_BM25,  edgecolor="white", linewidth=0.4, zorder=3)
    ax_in.bar(i + inset_width/2, em_local[i], inset_width,
              color=C_LOCAL, edgecolor="white", linewidth=0.4, zorder=3,
              yerr=[[err_lo[i]], [err_hi[i]]],
              error_kw=dict(ecolor="#555", capsize=3, capthick=1, elinewidth=1))

ax_in.set_xticks(range(n_models))
ax_in.set_xticklabels(["SC2\n7B", "SC2\n3B", "CL\n7B", "DS\n6.7B", "CQ\n7B"],
                       fontsize=8.5)
ax_in.set_ylim(73, 83)
ax_in.set_ylabel("Exact Match (%)", fontsize=10)
ax_in.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax_in.set_axisbelow(True)
ax_in.spines["top"].set_visible(False)
ax_in.spines["right"].set_visible(False)
ax_in.tick_params(axis="y", labelsize=9)
ax_in.set_title("(b) Zoomed: BM25 vs. Locality", fontsize=10, pad=8)

# Delta labels
for i, d in enumerate(deltas):
    ax_in.text(i + inset_width/2,
               em_local[i] + err_hi[i] + 0.2,
               f"+{d:.2f}pp", ha="center", va="bottom",
               fontsize=7.5, color="#2d6a2d", fontweight="bold")

# Legend for right panel
patch_bm25  = mpatches.Patch(color=C_BM25,  label="BM25 top-1")
patch_local = mpatches.Patch(color=C_LOCAL, label="Locality top-1 (ours)")
ax_in.legend(handles=[patch_bm25, patch_local],
             loc="lower right", fontsize=7.5, framealpha=0.9)

# ── Super-title ───────────────────────────────────────────────────────────────
fig.suptitle(
    "End-to-End Exact Match — 5 Model Architectures × 3 Retrieval Conditions  "
    "(RepoEval, 1,600 instances)",
    fontsize=11, y=1.01
)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout()
fig.savefig(OUT_PDF, format="pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_PNG, format="png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved:\n  {OUT_PDF}\n  {OUT_PNG}")
print(f"\nEM summary:")
for i, m in enumerate(["SC2-7B","SC2-3B","CL-7B","DS-6.7B","CQ-7B"]):
    print(f"  {m:12s}  prompt={em_prompt[i]:.2f}%  BM25={em_bm25[i]:.2f}%  "
          f"Locality={em_local[i]:.2f}%  delta=+{deltas[i]:.2f}pp")
print(f"\nSpread: prompt-only {em_prompt.max()-em_prompt.min():.2f}pp  "
      f"locality {em_local.max()-em_local.min():.2f}pp")
