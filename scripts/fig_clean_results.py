"""Figures for the five-condition RepoEval run (paper Section 5).

Usage: python scripts/fig_clean_results.py <per_example.csv> <output_dir>
Writes fig_conditions.pdf, fig_per_repo.pdf, fig_chunk_source.pdf,
fig_gain_loss.pdf and fig_strata.pdf.
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SAME = "#2a78d6"     # same-file chunk conditions (palette slot 1)
CROSS = "#eb6834"    # cross-file chunk (palette slot 2)
BASE = "#52514e"     # no-context baseline (neutral)
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"
OTHER_SAME = "#1baf7a"  # other same-file chunk (palette slot 3)
GAIN = "#52514e"
LOSS = "#b4b2ab"

CONDS = [  # (column, label, colour, marker)
    ("crossfile", "Cross-file chunk", CROSS, "D"),
    ("prompt_only", "Prompt-only (no chunk)", BASE, "o"),
    ("bm25", "BM25", SAME, "s"),
    ("locality", "Locality", SAME, "^"),
    ("nearest", "Nearest window", SAME, "v"),
]
REPO_LABEL = {
    "alibaba_FederatedScope": "FederatedScope",
    "awslabs_fortuna": "fortuna",
    "google_vizier": "vizier",
    "huggingface_diffusers": "diffusers",
    "huggingface_evaluate": "evaluate",
    "nerfstudio-project_nerfstudio": "nerfstudio",
    "opendilab_ACE": "ACE",
    "pytorch_rl": "rl",
}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def boot_ci(x, n_boot=10_000, seed=7):
    rng = random.Random(seed)
    n = len(x)
    s = sorted(sum(x[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return 100 * s[int(0.025 * n_boot)], 100 * s[int(0.975 * n_boot) - 1]


def fig_conditions(rows, out: Path):
    fig, ax = plt.subplots(figsize=(4.8, 2.0))
    ys = list(range(len(CONDS)))[::-1]
    for y, (col, label, colour, marker) in zip(ys, CONDS):
        x = [int(r[f"em_{col}"]) for r in rows]
        em = 100 * sum(x) / len(x)
        lo, hi = boot_ci(x)
        ax.plot([lo, hi], [y, y], color=colour, lw=2, solid_capstyle="round")
        ax.plot(em, y, marker=marker, color=colour, ms=7, mec="white", mew=1.2, ls="none")
        ax.text(54.5, y, f"{em:.2f}%", va="center", ha="right", fontsize=8, color=INK)
    po = 100 * sum(int(r["em_prompt_only"]) for r in rows) / len(rows)
    ax.axvline(po, color=BASE, lw=0.8, ls=(0, (3, 3)), zorder=0)
    ax.set_yticks(ys, [c[1] for c in CONDS])
    ax.set_xlim(20, 55)
    ax.set_xticks(range(20, 51, 5))
    ax.set_xlabel("Exact match (%), with 95% bootstrap CI")
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_per_repo(rows, out: Path):
    repos = sorted({r["repo"] for r in rows},
                   key=lambda k: sum(int(r["em_crossfile"]) - int(r["em_prompt_only"])
                                     for r in rows if r["repo"] == k))
    series = [("prompt_only", "Prompt-only", BASE, "o"),
              ("locality", "Locality", SAME, "^"),
              ("crossfile", "Cross-file chunk", CROSS, "D")]
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    for y, repo in enumerate(repos):
        sub = [r for r in rows if r["repo"] == repo]
        vals = {c: 100 * sum(int(r[f"em_{c}"]) for r in sub) / len(sub) for c, *_ in series}
        ax.plot([min(vals.values()), max(vals.values())], [y, y], color=GRID, lw=1.5, zorder=1)
        for col, label, colour, marker in series:
            ax.plot(vals[col], y, marker=marker, color=colour, ms=6.5, mec="white", mew=1.2,
                    ls="none", zorder=3, label=label if y == 0 else None)
    ax.set_yticks(range(len(repos)), [REPO_LABEL.get(r, r) for r in repos])
    ax.set_xlabel("Exact match (%), 200 instances per repository")
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False,
              handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def chunk_source(r, arm):
    if r[f"{arm}_top1_path"] != r["path"]:
        return "cross"
    same = (r[f"{arm}_top1_start"], r[f"{arm}_top1_end"]) == (r["nearest_start"], r["nearest_end"])
    return "nearest" if same else "other"


def fig_chunk_source(rows, out: Path):
    parts = [("nearest", "Nearest window", SAME),
             ("other", "Other same-file chunk", OTHER_SAME),
             ("cross", "Cross-file chunk", CROSS)]
    arms = [("locality", "Locality"), ("bm25", "BM25")]
    fig, ax = plt.subplots(figsize=(4.8, 1.75))
    for y, (arm, label) in enumerate(arms):
        left = 0
        for key, plabel, colour in parts:
            n = sum(1 for r in rows if chunk_source(r, arm) == key)
            ax.barh(y, n, left=left, height=0.55, color=colour, edgecolor="white", linewidth=1.5,
                    label=plabel if y == 0 else None)
            if n >= 120:
                ax.text(left + n / 2, y, f"{n:,}", ha="center", va="center", fontsize=8, color="white")
            else:
                ax.text(left + n + 15, y, f"{n}", ha="left", va="center", fontsize=8, color=INK)
            left += n
    ax.set_yticks(range(len(arms)), [a[1] for a in arms])
    ax.set_xlim(0, 1700)
    ax.set_xticks([0, 400, 800, 1200, 1600])
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=3, frameon=False,
              handlelength=1.0, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_gain_loss(rows, out: Path):
    conds = [("crossfile", "Cross-file chunk"), ("bm25", "BM25"),
             ("locality", "Locality"), ("nearest", "Nearest window")]
    fig, ax = plt.subplots(figsize=(4.8, 1.8))
    ys = list(range(len(conds)))[::-1]
    for y, (col, label) in zip(ys, conds):
        g = sum(1 for r in rows if r[f"em_{col}"] == "1" and r["em_prompt_only"] == "0")
        l = sum(1 for r in rows if r[f"em_{col}"] == "0" and r["em_prompt_only"] == "1")
        ax.barh(y, g, height=0.55, color=GAIN, label="Gains (correct only with the chunk)" if y == ys[0] else None)
        ax.barh(y, -l, height=0.55, color=LOSS, label="Losses (correct only without it)" if y == ys[0] else None)
        ax.text(g + 4, y, str(g), va="center", ha="left", fontsize=8, color=INK)
        ax.text(-l - 4, y, str(l), va="center", ha="right", fontsize=8, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(ys, [c[1] for c in conds])
    ax.set_xlim(-170, 230)
    ax.set_xticks([-150, -100, -50, 0, 50, 100, 150, 200])
    ax.set_xticklabels(["150", "100", "50", "0", "50", "100", "150", "200"])
    ax.set_xlabel("Instances that change outcome relative to prompt-only")
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.3), ncol=2, frameon=False,
              handlelength=1.0, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_strata(rows, out: Path):
    strata = [("same", "BM25 top-1 in the same file", SAME, "o"),
              ("cross", "BM25 top-1 in another file", CROSS, "D")]
    conds = [("prompt_only", "Prompt-only"), ("bm25", "BM25"), ("locality", "Locality"),
             ("nearest", "Nearest window"), ("crossfile", "Cross-file chunk")]
    fig, ax = plt.subplots(figsize=(4.8, 2.0))
    ys = list(range(len(conds)))[::-1]
    for key, label, colour, marker in strata:
        sub = [r for r in rows if (r["bm25_top1_path"] == r["path"]) == (key == "same")]
        xs = [100 * sum(int(r[f"em_{c}"]) for r in sub) / len(sub) for c, _ in conds]
        ax.plot(xs, ys, ls="none", marker=marker, color=colour, ms=7, mec="white", mew=1.2,
                label=f"{label} (n = {len(sub):,})", zorder=3)
    for y in ys:
        ax.axhline(y, color=GRID, lw=0.5, zorder=0)
    ax.set_yticks(ys, [c[1] for c in conds])
    ax.set_xlim(20, 75)
    ax.set_xlabel("Exact match (%)")
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, 1.28), ncol=2, frameon=False,
              handletextpad=0.3, columnspacing=1.0, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main(csv_path: str, out_dir: str):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8", newline="")))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_conditions(rows, out / "fig_conditions.pdf")
    fig_per_repo(rows, out / "fig_per_repo.pdf")
    fig_chunk_source(rows, out / "fig_chunk_source.pdf")
    fig_gain_loss(rows, out / "fig_gain_loss.pdf")
    fig_strata(rows, out / "fig_strata.pdf")
    print(f"wrote 5 figures to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
