"""
fig_completion_type_diverging.py
---------------------------------
Generates the diverging bar chart for the completion-type win/loss breakdown
(Discussion §5.2, aggregated over all 5 model runs, 8,000 comparisons total).

Win  = Locality correct AND BM25 wrong
Loss = BM25 correct AND Locality wrong

Data sourced directly from tab:typewins in discussion.tex.

Outputs: PDF (vector) and PNG (300 dpi) in paper/figures/.

Usage:
    python scripts/fig_completion_type_diverging.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_DIR  = os.path.join(os.path.dirname(REPO_ROOT), "paper")
OUT_DIR    = os.path.join(PAPER_DIR, "figures")
OUT_PDF   = os.path.join(OUT_DIR, "fig_completion_type_diverging.pdf")
OUT_PNG   = os.path.join(OUT_DIR, "fig_completion_type_diverging.png")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data from tab:typewins in discussion.tex ──────────────────────────────────
# 5 models × 1,600 instances = 8,000 total comparisons
categories   = ["import", "function_call", "control_flow",
                "assignment", "return", "other"]
counts       = [385,       1685,            630,
                2245,      300,             2550]
wins         = [40,        40,              15,
                22,        7,               42]
losses       = [0,         16,              5,
                8,         5,               17]

# Derived
win_rates = [w / c * 100 for w, c in zip(wins, counts)]
net       = [w - l for w, l in zip(wins, losses)]

# Display labels (category name + count)
labels = [
    f"import\n(n={counts[0]:,})",
    f"function_call\n(n={counts[1]:,})",
    f"control_flow\n(n={counts[2]:,})",
    f"assignment\n(n={counts[3]:,})",
    f"return\n(n={counts[4]:,})",
    f"other\n(n={counts[5]:,})",
]

# Sort by net gain descending for visual clarity
order = sorted(range(len(categories)), key=lambda i: net[i], reverse=True)
labels_sorted    = [labels[i]      for i in order]
wins_sorted      = [wins[i]        for i in order]
losses_sorted    = [losses[i]      for i in order]
net_sorted       = [net[i]         for i in order]
win_rates_sorted = [win_rates[i]   for i in order]
cats_sorted      = [categories[i]  for i in order]

# ── Colors ────────────────────────────────────────────────────────────────────
C_WIN  = "#2ca02c"   # green for wins (locality improves)
C_LOSS = "#d62728"   # red for losses (locality regresses)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.8))

y = np.arange(len(categories))
bar_height = 0.52

# Wins (extend right)
bars_win = ax.barh(y, wins_sorted, height=bar_height,
                   color=C_WIN, alpha=0.88, zorder=3,
                   label="Wins (Locality correct, BM25 wrong)")

# Losses (extend left, plotted as negative)
bars_loss = ax.barh(y, [-l for l in losses_sorted], height=bar_height,
                    color=C_LOSS, alpha=0.75, zorder=3,
                    label="Losses (BM25 correct, Locality wrong)")

# ── Centre line ───────────────────────────────────────────────────────────────
ax.axvline(0, color="black", linewidth=1.2, zorder=4)

# ── Data labels ───────────────────────────────────────────────────────────────
for i, (w, l, wr) in enumerate(zip(wins_sorted, losses_sorted, win_rates_sorted)):
    # Win label (right side)
    x_win = w + 0.4
    ax.text(x_win, y[i], f"{w}  ({wr:.1f}%)",
            ha="left", va="center", fontsize=9, color=C_WIN, fontweight="bold")
    # Loss label (left side) — only if > 0
    if l > 0:
        ax.text(-(l + 0.4), y[i], f"−{l}",
                ha="right", va="center", fontsize=9, color=C_LOSS)

# ── Special annotation for import row (zero losses) ──────────────────────────
import_idx = cats_sorted.index("import")
ax.annotate("Zero\nlosses",
            xy=(-0.15, y[import_idx]),
            xytext=(-12, y[import_idx]),
            fontsize=8, color=C_LOSS, ha="right", va="center",
            arrowprops=dict(arrowstyle="-|>", color=C_LOSS,
                            lw=1.2, mutation_scale=10))

# ── Net gain labels (right margin) ───────────────────────────────────────────
ax_max = max(wins_sorted) + 8
for i, n in enumerate(net_sorted):
    ax.text(ax_max + 3.5, y[i], f"net: +{n}",
            ha="left", va="center", fontsize=8.5, color="#333",
            style="italic")

# ── Y-axis ────────────────────────────────────────────────────────────────────
ax.set_yticks(y)
ax.set_yticklabels(labels_sorted, fontsize=10)
ax.invert_yaxis()   # highest net gain at top

# ── X-axis ────────────────────────────────────────────────────────────────────
max_x = max(wins_sorted) + 10
ax.set_xlim(-max(losses_sorted) - 5, max_x + 18)
ax.set_xlabel("Number of instances", fontsize=11)

# Remove default x-tick labels (replace with left/right annotations)
xticks = [-15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]
ax.set_xticks([t for t in xticks if t >= ax.get_xlim()[0]])
ax.set_xticklabels([str(abs(t)) for t in xticks if t >= ax.get_xlim()[0]],
                   fontsize=9)

ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Direction labels
ax.text(-max(losses_sorted) - 4.5, -0.7, "← Losses\n  (BM25 wins)",
        ha="left", va="top", fontsize=8.5, color=C_LOSS, style="italic")
ax.text(1.5, -0.7, "Wins →\n(Locality wins)",
        ha="left", va="top", fontsize=8.5, color=C_WIN, style="italic")

# ── Title and legend ──────────────────────────────────────────────────────────
ax.set_title(
    "Completion-Type Win/Loss Breakdown — Locality Reranking vs. BM25\n"
    "(aggregated over 5 models × 1,600 instances = 8,000 comparisons; "
    "win rate = wins / category count)",
    fontsize=10.5, pad=10
)

patch_win  = mpatches.Patch(color=C_WIN,  alpha=0.88,
                             label="Win: Locality correct, BM25 wrong")
patch_loss = mpatches.Patch(color=C_LOSS, alpha=0.75,
                             label="Loss: BM25 correct, Locality wrong")
ax.legend(handles=[patch_win, patch_loss],
          loc="lower right", fontsize=8.5, framealpha=0.9)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
fig.savefig(OUT_PDF, format="pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_PNG, format="png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved:\n  {OUT_PDF}\n  {OUT_PNG}")
print(f"\nWin/Loss summary (sorted by net):")
for i in order:
    print(f"  {categories[i]:15s}  count={counts[i]:5,}  "
          f"wins={wins[i]:3d}  losses={losses[i]:3d}  "
          f"net={net[i]:+3d}  win_rate={win_rates[i]:.1f}%")
