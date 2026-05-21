"""
fig_venn_outcome.py
--------------------
Generates the Venn diagram showing overlap between BM25-correct and
Locality-correct instance sets (StarCoder2-7B, 1,600 instances).

Data from tab:outcome in discussion.tex:
  BM25 correct, Locality correct : 1240
  BM25 correct, Locality wrong   :   11
  BM25 wrong,   Locality correct :   30
  BM25 wrong,   Locality wrong   :  319   (outside both circles)

Draws proportional-area circles with overlap annotation.
Uses matplotlib patches only (no external Venn library needed).

Outputs: PDF (vector) and PNG (300 dpi) in paper/figures/.

Usage:
    python scripts/fig_venn_outcome.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_DIR  = os.path.join(os.path.dirname(REPO_ROOT), "paper")
OUT_DIR    = os.path.join(PAPER_DIR, "figures")
OUT_PDF   = os.path.join(OUT_DIR, "fig_venn_outcome.pdf")
OUT_PNG   = os.path.join(OUT_DIR, "fig_venn_outcome.png")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Outcome matrix (StarCoder2-7B, 1,600 instances) ──────────────────────────
bm25_only   = 11     # BM25 correct, Locality wrong  (regression)
shared      = 1240   # both correct
local_only  = 30     # Locality correct, BM25 wrong  (gain)
neither     = 319    # both wrong

total       = 1600
bm25_total  = bm25_only + shared    # 1251
local_total = local_only + shared   # 1270

# ── Fixed layout: circles at fixed positions (not area-proportional) ──────────
# Overlap is 97% of both sets so proportional layout is unreadable.
# Use equal-radius circles with clear overlap for legibility.
r_bm25 = 0.50
r_loc  = 0.50

cx_bm25 = -0.26
cx_loc  =  0.26
cy      = 0.0

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7.0))
ax.set_aspect("equal")
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.00, 1.15)
ax.axis("off")

# Draw circles
circle_bm25 = Circle((cx_bm25, cy), r_bm25,
                     facecolor="#4c72b0", alpha=0.30,
                     edgecolor="#1f4e9e", linewidth=2.0)
circle_loc  = Circle((cx_loc,  cy), r_loc,
                     facecolor="#dd8452", alpha=0.30,
                     edgecolor="#a84500", linewidth=2.0)
ax.add_patch(circle_bm25)
ax.add_patch(circle_loc)

# ── Circle title labels (above circles) ──────────────────────────────────────
ax.text(cx_bm25 - 0.10, cy + r_bm25 + 0.18,
        f"BM25 correct\nn = {bm25_total:,}",
        ha="center", va="bottom", fontsize=10.5, fontweight="bold",
        color="#1f4e9e")
ax.text(cx_loc + 0.10, cy + r_loc + 0.18,
        f"Locality correct\nn = {local_total:,}",
        ha="center", va="bottom", fontsize=10.5, fontweight="bold",
        color="#a84500")

# ── Region labels ─────────────────────────────────────────────────────────────
# BM25-only region: x = cx_bm25 - (r - overlap/2) ≈ -0.63
x_bm25only = cx_bm25 - 0.24
ax.text(x_bm25only, cy + 0.10, f"{bm25_only}",
        ha="center", va="center", fontsize=22, fontweight="bold", color="#1f4e9e")
ax.text(x_bm25only, cy - 0.11, "BM25 only",
        ha="center", va="center", fontsize=8.5, color="#1f4e9e", style="italic")
ax.text(x_bm25only, cy - 0.20, "(regressions)",
        ha="center", va="center", fontsize=8, color="#1f4e9e", style="italic")
ax.text(x_bm25only, cy - 0.29, f"({bm25_only/total*100:.2f}%)",
        ha="center", va="center", fontsize=8, color="#555")

# Shared region: centre x = 0
ax.text(0, cy + 0.10, f"{shared:,}",
        ha="center", va="center", fontsize=22, fontweight="bold", color="#333")
ax.text(0, cy - 0.14, "Both correct",
        ha="center", va="center", fontsize=9, color="#444")
ax.text(0, cy - 0.27, f"({shared/total*100:.1f}%)",
        ha="center", va="center", fontsize=8.5, color="#555")

# Locality-only region: x = cx_loc + (r - overlap/2) ≈ +0.63
x_loconly = cx_loc + 0.24
ax.text(x_loconly, cy + 0.10, f"{local_only}",
        ha="center", va="center", fontsize=22, fontweight="bold", color="#a84500")
ax.text(x_loconly, cy - 0.11, "Locality only",
        ha="center", va="center", fontsize=8.5, color="#a84500", style="italic")
ax.text(x_loconly, cy - 0.20, "(gains)",
        ha="center", va="center", fontsize=8, color="#a84500", style="italic")
ax.text(x_loconly, cy - 0.29, f"({local_only/total*100:.2f}%)",
        ha="center", va="center", fontsize=8, color="#555")

# ── "Neither" box (outside both circles) ────────────────────────────────────
ax.text(0, cy - r_bm25 - 0.18,
        f"Neither correct: {neither} ({neither/total*100:.1f}%)",
        ha="center", va="top", fontsize=9, color="#555",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                  edgecolor="#ccc", linewidth=0.8))

# ── Net gain annotation ───────────────────────────────────────────────────────
net = local_only - bm25_only   # 30 - 11 = 19
ax.annotate(
    f"Net gain: +{net} instances\n(+{net/total*100:.2f} pp EM)",
    xy=(x_loconly, cy + 0.14),
    xytext=(0.72, 0.88),
    xycoords="data", textcoords="data",
    fontsize=9.5, color="#a84500", fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="-|>", color="#a84500", lw=1.3,
                    connectionstyle="arc3,rad=-0.2")
)

# ── Regression rate annotation ────────────────────────────────────────────────
ax.annotate(
    f"Regression rate:\n{bm25_only}/{bm25_total} = {bm25_only/bm25_total*100:.1f}%",
    xy=(x_bm25only, cy + 0.14),
    xytext=(-0.72, 0.88),
    xycoords="data", textcoords="data",
    fontsize=9, color="#1f4e9e", ha="center",
    arrowprops=dict(arrowstyle="-|>", color="#1f4e9e", lw=1.3,
                    connectionstyle="arc3,rad=0.2")
)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title(
    "Outcome Sets: BM25 top-1 vs. Locality top-1  "
    "(StarCoder2-7B, RepoEval, 1,600 instances)\n"
    "Locality gains 30 instances BM25 misses; loses only 11 BM25 had correct",
    fontsize=10.5, pad=14, y=1.02
)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
fig.savefig(OUT_PDF, format="pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_PNG, format="png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved:\n  {OUT_PDF}\n  {OUT_PNG}")
print(f"\nOutcome matrix (StarCoder2-7B, n={total}):")
print(f"  Both correct        : {shared:4d}  ({shared/total*100:.1f}%)")
print(f"  BM25 only (regress) : {bm25_only:4d}  ({bm25_only/total*100:.2f}%) "
      f"— regression rate {bm25_only/bm25_total*100:.1f}% of BM25 correct")
print(f"  Locality only (gain): {local_only:4d}  ({local_only/total*100:.2f}%)")
print(f"  Neither correct     : {neither:4d}  ({neither/total*100:.1f}%)")
print(f"  Net gain            : +{net} instances  (+{net/total*100:.2f} pp EM)")
