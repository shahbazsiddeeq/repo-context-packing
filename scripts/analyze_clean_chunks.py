"""Secondary analyses of the five-condition run (paper Sections 5.4-5.6).

Inputs: the per-instance CSV of the run and the JSONL written by
scripts/extract_prompt_chunks.py. Writes a JSON file with:
  - edit similarity (ES) per condition and paired bootstrap CIs of ES differences;
  - redundancy of each prompt chunk with the prefix (share of chunk lines that
    already occur in the prefix, comparing lines with all whitespace removed);
  - novel-identifier coverage: identifiers of the reference line that are absent
    from the prefix, and whether the prompt chunk contains one of them;
  - exact match by kind of target line;
  - copying behaviour: predictions that repeat a chunk line or a prefix line, and
    predictions that are comment lines.

Usage: python scripts/analyze_clean_chunks.py <per_example.csv> <prompt_chunks.jsonl> <out.json>
"""
from __future__ import annotations

import ast
import csv
import json
import re
import sys

import numpy as np
from scipy.stats import binomtest

ARMS = ["prompt_only", "bm25", "locality", "nearest", "crossfile"]
CHUNK_ARMS = ARMS[1:]
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
KEYWORDS = {
    "False", "None", "True", "and", "as", "assert", "async", "await", "break",
    "class", "continue", "def", "del", "elif", "else", "except", "finally", "for",
    "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or",
    "pass", "raise", "return", "try", "while", "with", "yield", "self", "cls",
}
SEED = 7
ITERS = 10_000


def norm(line):
    """Line key that ignores all whitespace (RepoEval prefixes are detokenized)."""
    return "".join(line.split())


def idents(text):
    return {t for t in IDENT.findall(text) if t not in KEYWORDS}


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def edit_sim(pred, gold):
    p, g = pred.strip(), gold.strip()
    if not p and not g:
        return 1.0
    return 1.0 - levenshtein(p, g) / max(len(p), len(g))


def _parse(s):
    for src in (s, s + "\n    pass", s.rstrip(":") ):
        try:
            return ast.parse(src).body
        except SyntaxError:
            continue
    return None


def line_kind(gold):
    """Kind of the reference line, from the line alone.

    Continuation lines (arguments, elements, closing brackets) are recognised by
    their first or last character; other lines are parsed as Python and classified
    by the AST node; lines that do not parse (mostly docstring text) form their own
    group.
    """
    s = gold.strip()
    if s.startswith(("import ", "from ")):
        return "import"
    if s.startswith(("@", "def ", "class ", "async def ")):
        return "definition"
    if s.endswith((",", "(", "[", "{")) or s[:1] in ")]}":
        return "argument/element"
    if re.match(r"(elif|else|except|finally)\b", s):
        return "control flow"
    body = _parse(s)
    if not body:
        return "not parsable"
    node = body[0]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return "definition"
    if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith,
                         ast.Try, ast.Assert, ast.Break, ast.Continue, ast.Pass)):
        return "control flow"
    if isinstance(node, (ast.Return, ast.Raise)) or (
            isinstance(node, ast.Expr) and isinstance(node.value, (ast.Yield, ast.YieldFrom))):
        return "return/raise"
    if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        return "assignment"
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        return "call"
    return "other"


def mcnemar(a, b):
    gains = int(((a == 0) & (b == 1)).sum())
    losses = int(((a == 1) & (b == 0)).sum())
    p = 1.0 if gains + losses == 0 else binomtest(min(gains, losses), gains + losses, 0.5).pvalue
    return gains, losses, float(p)


def boot_ci(diff):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(diff), size=(ITERS, len(diff)))
    means = diff[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    csv_path, jsonl_path, out_path = sys.argv[1:4]
    with open(csv_path, encoding="utf-8") as f:
        rows = {r["task_id"]: r for r in csv.DictReader(f)}
    with open(jsonl_path, encoding="utf-8") as f:
        chunks = {c["task_id"]: c for c in map(json.loads, f)}
    ids = [t for t in rows if t in chunks]
    n = len(ids)
    em = {a: np.array([int(rows[t][f"em_{a}"]) for t in ids]) for a in ARMS}
    es = {a: np.array([edit_sim(rows[t][f"pred_{a}"], rows[t]["gold_line"]) for t in ids])
          for a in ARMS}
    out = {"n": n}

    # Edit similarity.
    out["es_mean"] = {a: float(es[a].mean() * 100) for a in ARMS}
    out["es_paired"] = {}
    for a, b in [("bm25", "prompt_only"), ("locality", "prompt_only"),
                 ("nearest", "prompt_only"), ("crossfile", "prompt_only"),
                 ("locality", "bm25"), ("locality", "nearest"), ("crossfile", "bm25")]:
        d = (es[a] - es[b]) * 100
        out["es_paired"][f"{a}_vs_{b}"] = {"delta": float(d.mean()), "ci": boot_ci(d)}

    # Redundancy with the prefix and novel-identifier coverage.
    red, cover, kinds = {a: [] for a in CHUNK_ARMS}, {a: [] for a in CHUNK_ARMS}, []
    novel_any = []
    copy_chunk = {a: [] for a in CHUNK_ARMS}
    copy_prefix = {a: [] for a in ARMS}
    comment = {a: [] for a in ARMS}
    for t in ids:
        r, c = rows[t], chunks[t]
        prefix_lines = {norm(ln) for ln in c["prefix"].splitlines() if ln.strip()}
        novel = idents(r["gold_line"]) - idents(c["prefix"])
        novel_any.append(bool(novel))
        kinds.append(line_kind(r["gold_line"]))
        for a in CHUNK_ARMS:
            ch = c[a] if a != "nearest" or c[a] else None
            text = ch["text"] if ch else ""
            lines = [norm(ln) for ln in text.splitlines() if ln.strip()]
            red[a].append(sum(ln in prefix_lines for ln in lines) / len(lines) if lines else 1.0)
            cover[a].append(bool(novel & idents(text)))
            pred = norm(r[f"pred_{a}"])
            copy_chunk[a].append(bool(pred) and pred in set(lines))
        for a in ARMS:
            pred = norm(r[f"pred_{a}"])
            copy_prefix[a].append(bool(pred) and pred in prefix_lines)
            comment[a].append(pred.startswith("#"))
    novel_any = np.array(novel_any)
    out["novel_identifier_instances"] = int(novel_any.sum())
    out["redundancy"] = {}
    out["coverage"] = {}
    for a in CHUNK_ARMS:
        rd = np.array(red[a])
        cv = np.array(cover[a])
        g_cov, l_cov, p_cov = mcnemar(em["prompt_only"][cv], em[a][cv])
        g_unc, l_unc, p_unc = mcnemar(em["prompt_only"][~cv], em[a][~cv])
        out["redundancy"][a] = {
            "mean_share_in_prefix": float(rd.mean() * 100),
            "fully_in_prefix": int((rd >= 1.0).sum()),
        }
        out["coverage"][a] = {
            "covers_novel": int(cv.sum()),
            "covers_share_of_novel_instances": float(cv.sum() / novel_any.sum() * 100),
            "em_prompt_only_when_covers": float(em["prompt_only"][cv].mean() * 100) if cv.any() else None,
            "em_arm_when_covers": float(em[a][cv].mean() * 100) if cv.any() else None,
            "gains_losses_p_when_covers": [g_cov, l_cov, p_cov],
            "em_prompt_only_when_not": float(em["prompt_only"][~cv].mean() * 100),
            "em_arm_when_not": float(em[a][~cv].mean() * 100),
            "gains_losses_p_when_not": [g_unc, l_unc, p_unc],
        }
    # EM by whether the reference line has any identifier missing from the prefix.
    out["by_novelty"] = {
        "with_novel": {a: float(em[a][novel_any].mean() * 100) for a in ARMS},
        "without_novel": {a: float(em[a][~novel_any].mean() * 100) for a in ARMS},
    }

    # Exact match by kind of target line.
    kinds = np.array(kinds)
    out["by_kind"] = {}
    for k in sorted(set(kinds)):
        m = kinds == k
        g, l, p = mcnemar(em["prompt_only"][m], em["crossfile"][m])
        g2, l2, p2 = mcnemar(em["prompt_only"][m], em["bm25"][m])
        out["by_kind"][k] = {
            "n": int(m.sum()),
            "em": {a: float(em[a][m].mean() * 100) for a in ARMS},
            "crossfile_vs_prompt_only": [g, l, p],
            "bm25_vs_prompt_only": [g2, l2, p2],
        }

    # Copying behaviour.
    out["copy"] = {}
    for a in ARMS:
        cp = np.array(copy_prefix[a])
        cm = np.array(comment[a])
        entry = {
            "repeats_prefix_line": int(cp.sum()),
            "repeats_prefix_line_correct": int((cp & (em[a] == 1)).sum()),
            "comment_line": int(cm.sum()),
        }
        if a in CHUNK_ARMS:
            cc = np.array(copy_chunk[a])
            entry["repeats_chunk_line"] = int(cc.sum())
            entry["repeats_chunk_line_correct"] = int((cc & (em[a] == 1)).sum())
        out["copy"][a] = entry

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
