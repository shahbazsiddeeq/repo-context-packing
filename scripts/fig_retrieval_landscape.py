"""
fig_retrieval_landscape.py
---------------------------
Generates the retrieval quality landscape figure: MRR, Recall@1, and Recall@5
for all 6 retrieval conditions across RQ1, RQ3, and RQ5.

Consolidates three tables (table_chunking, table_retrieval, table_dense)
into one figure sorted by MRR. Shows where locality reranking sits relative
to all baselines across both retrieval backends.

All values are retrieval metrics (not EM), so cross-condition comparison is valid.

Data sources:
  - outputs/chunking_comparison_1600/chunking_comparison_summary.csv
  - outputs/ablation_beta_lambda/ablation_results.csv  (BM25 baseline + paper setting)
  - outputs/dense_retrieval/dense_retrieval_summary.json

Outputs: PDF (vector) and PNG (300 dpi) in paper/figures/.

Usage:
    python scripts/fig_retrieval_landscape.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_DIR  = os.path.join(os.path.dirname(REPO_ROOT), "paper")
OUT_DIR    = os.path.join(PAPER_DIR, "figures")
OUT_PDF   = os.path.join(OUT_DIR, "fig_retrieval_landscape.pdf")
OUT_PNG   = os.path.join(OUT_DIR, "fig_retrieval_landscape.png")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data — all values from verified output files ──────────────────────────────
# (condition, MRR, Recall@1, Recall@5, backend, is_ours, note)
conditions = [
    # label                           MRR       R@1       R@5      backend  ours  note
    ("Hybrid BM25",                   0.5540,   0.3800,   0.7944,  "bm25",  False, ""),
    ("Dense\n(UniXcoder)",            0.6183,   0.4863,   0.7875,  "dense", False, ""),
    ("AST-based BM25",                0.6595,   0.5453,   0.8148,  "bm25",  False, "†"),
    ("Fixed BM25",                    0.7060,   0.5831,   0.8556,  "bm25",  False, ""),
    ("Fixed BM25\n+ Locality (ours)", 0.7394,   0.6206,   0.8831,  "bm25",  True,  ""),
    ("Dense + Locality (ours)\n(UniXcoder)", 0.7719, 0.7338, 0.8225, "dense", True, ""),
]

# Sort by MRR ascending (lowest at bottom → Dense+Locality at top)
conditions.sort(key=lambda x: x[1])

labels    = [c[0] for c in conditions]
mrr       = np.array([c[1] for c in conditions])
recall1   = np.array([c[2] for c in conditions])
recall5   = np.array([c[3] for c in conditions])
backends  = [c[4] for c in conditions]
is_ours   = [c[5] for c in conditions]
notes     = [c[6] for c in conditions]

n = len(conditions)
y = np.arange(n)

# ── Color scheme ──────────────────────────────────────────────────────────────
# BM25 backend: blue family; Dense backend: orange family
# "Ours" variants: bold/dark; baselines: lighter
def bar_color(backend, ours):
    if backend == "bm25":
        return "#1f77b4" if ours else "#aec7e8"   # dark blue vs light blue
    else:
        return "#d62728" if ours else "#f5a9a9"   # dark red vs light red

bar_height = 0.22   # height per metric bar
group_pad  = 0.1    # extra gap between groups

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))

# Offsets within each group: MRR (top), R@1 (middle), R@5 (bottom)
offsets = [bar_height, 0, -bar_height]
metric_labels = ["MRR", "Recall@1", "Recall@5"]
metric_colors = ["#333333", "#555555", "#888888"]  # annotation text colors
metric_values = [mrr, recall1, recall5]
hatch_patterns = ["", "//", ".."]

for mi, (offset, values, mlabel, hatch) in enumerate(
        zip(offsets, metric_values, metric_labels, hatch_patterns)):
    for i in range(n):
        color = bar_color(backends[i], is_ours[i])
        edge  = "#222" if is_ours[i] else "#888"
        lw    = 1.4 if is_ours[i] else 0.6
        ax.barh(y[i] + offset, values[i], height=bar_height,
                color=color, edgecolor=edge, linewidth=lw,
                hatch=hatch, alpha=0.90, zorder=3)

    # Metric legend patch
    ax.barh(-2, 0, height=bar_height,
            color="#bbbbbb", edgecolor="#888", linewidth=0.6,
            hatch=hatch, alpha=0.90, label=mlabel, zorder=1)

# ── Value annotations on MRR bars only ───────────────────────────────────────
for i in range(n):
    ax.text(mrr[i] + 0.004, y[i] + offsets[0],
            f"{mrr[i]:.4f}",
            va="center", ha="left", fontsize=8,
            fontweight="bold" if is_ours[i] else "normal",
            color="#1f77b4" if backends[i] == "bm25" else "#d62728")
    ax.text(recall1[i] + 0.004, y[i] + offsets[1],
            f"{recall1[i]:.4f}",
            va="center", ha="left", fontsize=7.5, color="#444")
    ax.text(recall5[i] + 0.004, y[i] + offsets[2],
            f"{recall5[i]:.4f}",
            va="center", ha="left", fontsize=7.5, color="#444")

# ── "Ours" markers ────────────────────────────────────────────────────────────
for i in range(n):
    if is_ours[i]:
        ax.text(-0.005, y[i], "★",
                va="center", ha="right", fontsize=11,
                color="#1f77b4" if backends[i] == "bm25" else "#d62728")

# ── AST footnote marker ───────────────────────────────────────────────────────
for i, note in enumerate(notes):
    if note:
        ax.text(mrr[i] + 0.004, y[i] + offsets[0] + 0.14,
                note, va="bottom", ha="left", fontsize=8, color="#666")

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlim(0.30, 0.92)
ax.set_xlabel("Retrieval metric value", fontsize=11)
ax.xaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Bold y-tick labels for "ours" conditions
for i, (label, tick) in enumerate(zip(labels, ax.get_yticklabels())):
    if is_ours[i]:
        tick.set_fontweight("bold")

# ── Reference line: BM25 baseline MRR ────────────────────────────────────────
bm25_mrr = 0.7060
ax.axvline(bm25_mrr, color="#aec7e8", linewidth=1.2, linestyle=":",
           alpha=0.8, zorder=2)
ax.text(bm25_mrr + 0.002, -0.7, "BM25\nbaseline", fontsize=7.5,
        color="#aec7e8", va="top", ha="left")

# ── Backend legend ────────────────────────────────────────────────────────────
p_bm25_base  = mpatches.Patch(color="#aec7e8", label="BM25 backend (baseline)")
p_bm25_ours  = mpatches.Patch(color="#1f77b4", label="BM25 + Locality ★ (ours)")
p_dense_base = mpatches.Patch(color="#f5a9a9", label="Dense backend (baseline)")
p_dense_ours = mpatches.Patch(color="#d62728", label="Dense + Locality ★ (ours)")

# Metric hatch legend
p_mrr  = mpatches.Patch(facecolor="#bbbbbb", edgecolor="#444", linewidth=0.8,
                         hatch="",   label="MRR (value labelled)")
p_r1   = mpatches.Patch(facecolor="#bbbbbb", edgecolor="#444", linewidth=0.8,
                         hatch="//", label="Recall@1")
p_r5   = mpatches.Patch(facecolor="#bbbbbb", edgecolor="#444", linewidth=0.8,
                         hatch="..", label="Recall@5")

leg1 = ax.legend(handles=[p_bm25_base, p_bm25_ours, p_dense_base, p_dense_ours],
                 loc="lower right", fontsize=8.5, framealpha=0.92,
                 title="Backend", title_fontsize=8.5,
                 bbox_to_anchor=(1.0, 0.0))
ax.add_artist(leg1)
ax.legend(handles=[p_mrr, p_r1, p_r5],
          loc="lower right", fontsize=8.5, framealpha=0.92,
          title="Metric (bar pattern)", title_fontsize=8.5,
          bbox_to_anchor=(0.62, 0.0))

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title(
    "Retrieval Quality Landscape: MRR, Recall@1, and Recall@5 Across All Conditions\n"
    "(RepoEval, 1,600 instances; sorted by MRR; "
    "★ = our method; "
    "† AST-based: 34/1,600 instances with no covering chunk)",
    fontsize=10, pad=10
)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
fig.savefig(OUT_PDF, format="pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_PNG, format="png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved:\n  {OUT_PDF}\n  {OUT_PNG}")
print(f"\nConditions sorted by MRR:")
for c in sorted(conditions, key=lambda x: x[1], reverse=True):
    ours = "★ ours" if c[5] else "     "
    print(f"  {ours}  {c[0]:40s}  MRR={c[1]:.4f}  R@1={c[2]:.4f}  R@5={c[3]:.4f}")
