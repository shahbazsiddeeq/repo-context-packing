"""
fig_ablation_heatmap.py
-----------------------
Generates the hyperparameter sensitivity heatmap (Figure: RQ3 ablation).

Grid: 7 beta values x 7 lambda values -> MRR on RepoEval (1,600 instances).
Outputs: PDF (vector) and PNG (300 dpi) in paper/figures/.

Usage:
    python scripts/fig_ablation_heatmap.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "outputs", "ablation_beta_lambda",
                         "ablation_results.csv")
OUT_DIR   = os.path.join(REPO_ROOT, "paper", "figures")
OUT_PDF   = os.path.join(OUT_DIR, "fig_ablation_heatmap.pdf")
OUT_PNG   = os.path.join(OUT_DIR, "fig_ablation_heatmap.png")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

# BM25 baseline row (beta=0, lambda=0)
bm25_row = df[df["label"] == "BM25-baseline"].iloc[0]
BM25_MRR = bm25_row["MRR"]          # 0.706025

# Paper setting (beta=150, lambda=0.2)
paper_row = df[df["is_paper_setting"] == True].iloc[0]
PAPER_BETA   = int(paper_row["beta"])    # 150
PAPER_LAMBDA = paper_row["lambda"]       # 0.2

# Grid: 49 combinations only
grid_df = df[df["beta"] > 0].copy()

beta_vals   = sorted(grid_df["beta"].unique())     # [25, 50, 100, 150, 200, 300, 500]
lambda_vals = sorted(grid_df["lambda"].unique())   # [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

# Build MRR matrix: rows = beta (y-axis, reversed so largest at top),
#                   cols = lambda (x-axis)
mrr_matrix = np.zeros((len(beta_vals), len(lambda_vals)))
for i, b in enumerate(beta_vals):
    for j, l in enumerate(lambda_vals):
        val = grid_df[(grid_df["beta"] == b) & (grid_df["lambda"] == l)]["MRR"]
        mrr_matrix[i, j] = float(val.iloc[0])

# Flip rows so largest beta is at top
mrr_matrix = mrr_matrix[::-1]
beta_labels = [str(int(b)) for b in reversed(beta_vals)]

lambda_labels = []
for l in lambda_vals:
    if l == 0.0:
        lambda_labels.append("0")
    elif l == int(l):
        lambda_labels.append(str(int(l)))
    else:
        lambda_labels.append(str(l))

# Paper setting index in the flipped matrix
paper_row_idx = len(beta_vals) - 1 - beta_vals.index(PAPER_BETA)
paper_col_idx = lambda_vals.index(PAPER_LAMBDA)

# ── Colormap: diverging from BM25 baseline upward ────────────────────────────
vmin = BM25_MRR        # 0.7060  (baseline lower anchor)
vmax = mrr_matrix.max()  # 0.7466

# Use a perceptually uniform sequential colormap
cmap = plt.cm.YlOrRd   # yellow (low) -> red (high), easy to read in B&W too

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 5.2))

norm  = Normalize(vmin=vmin, vmax=vmax)
im    = ax.imshow(mrr_matrix, cmap=cmap, norm=norm, aspect="auto")

# ── Cell annotations ─────────────────────────────────────────────────────────
for i in range(len(beta_vals)):
    for j in range(len(lambda_vals)):
        val  = mrr_matrix[i, j]
        # Use black text on light cells, white on dark
        brightness = (val - vmin) / (vmax - vmin)
        text_color = "white" if brightness > 0.65 else "black"
        ax.text(j, i, f"{val:.4f}",
                ha="center", va="center",
                fontsize=7.5, color=text_color, fontweight="normal")

# ── Mark paper setting (beta=150, lambda=0.2) ────────────────────────────────
rect = mpatches.FancyBboxPatch(
    (paper_col_idx - 0.48, paper_row_idx - 0.48),
    0.96, 0.96,
    boxstyle="square,pad=0",
    linewidth=2.8, edgecolor="#1a1aff", facecolor="none",
    zorder=5
)
ax.add_patch(rect)
# Star marker above the paper-setting cell
ax.plot(paper_col_idx, paper_row_idx - 0.56,
        marker="*", markersize=12, color="#1a1aff",
        markeredgecolor="#1a1aff", zorder=6, linestyle="none")

# ── Axes labels ───────────────────────────────────────────────────────────────
ax.set_xticks(range(len(lambda_vals)))
ax.set_xticklabels(lambda_labels, fontsize=10)
ax.set_yticks(range(len(beta_vals)))
ax.set_yticklabels(beta_labels, fontsize=10)

ax.set_xlabel(r"Line-distance weight  $\lambda$", fontsize=11, labelpad=6)
ax.set_ylabel(r"Same-file bonus  $\beta$", fontsize=11, labelpad=6)

# ── Colorbar ─────────────────────────────────────────────────────────────────
cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                    fraction=0.046, pad=0.02)
cbar.set_label("MRR", fontsize=10, labelpad=6)
cbar.ax.tick_params(labelsize=9)

# Mark BM25 baseline on colorbar
cbar.ax.axhline(y=(BM25_MRR - vmin) / (vmax - vmin),
                color="#2ca02c", linewidth=1.8, linestyle="--")
cbar.ax.text(1.12, (BM25_MRR - vmin) / (vmax - vmin),
             f"BM25={BM25_MRR:.4f}",
             transform=cbar.ax.transAxes,
             va="center", ha="left", fontsize=7.5, color="#2ca02c")

# ── Title and legend note ─────────────────────────────────────────────────────
ax.set_title(
    "Hyperparameter Sensitivity: MRR over 49 $\\beta$–$\\lambda$ Combinations\n"
    r"(BM25 baseline MRR = 0.7060; all 49 cells above baseline)",
    fontsize=10.5, pad=10
)

# Legend for the blue box
legend_patch = mpatches.Patch(facecolor="none", edgecolor="#1a1aff",
                               linewidth=2.5,
                               label=r"Paper setting ($\beta$=150, $\lambda$=0.2)")
ax.legend(handles=[legend_patch], loc="lower left",
          fontsize=8.5, framealpha=0.9,
          bbox_to_anchor=(0.0, -0.18), ncol=1)

plt.tight_layout(rect=[0, 0.05, 1, 1])

# ── Save ──────────────────────────────────────────────────────────────────────
fig.savefig(OUT_PDF, format="pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_PNG, format="png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved:\n  {OUT_PDF}\n  {OUT_PNG}")
print(f"\nGrid stats:")
print(f"  BM25 baseline MRR : {BM25_MRR:.4f}")
print(f"  Grid min MRR      : {mrr_matrix.min():.4f}  (beta={beta_vals[len(beta_vals)-1-np.unravel_index(mrr_matrix.argmin(), mrr_matrix.shape)[0]]:.0f}, lambda={lambda_vals[np.unravel_index(mrr_matrix.argmin(), mrr_matrix.shape)[1]]})")
print(f"  Grid max MRR      : {mrr_matrix.max():.4f}  (beta={beta_vals[len(beta_vals)-1-np.unravel_index(mrr_matrix.argmax(), mrr_matrix.shape)[0]]:.0f}, lambda={lambda_vals[np.unravel_index(mrr_matrix.argmax(), mrr_matrix.shape)[1]]})")
print(f"  Paper setting MRR : {mrr_matrix[paper_row_idx, paper_col_idx]:.4f}  (beta={PAPER_BETA}, lambda={PAPER_LAMBDA})")
