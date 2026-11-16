#!/usr/bin/env bash
# Clean (truncated-index) RepoEval run with five conditions: prompt-only, BM25,
# locality, nearest preceding window, and top-ranked cross-file chunk.
# Run from the repository root. Extra arguments (e.g. --start_row/--end_row,
# --output_dir) are passed through. Resumes from its checkpoint if restarted.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD}"
OUT="outputs/CR_clean_repocoder_starcoder2_7b_1600"
python -u scripts/run_repoeval_completion_model.py \
  --model_name bigcode/starcoder2-7b --max_rows 1600 \
  --output_dir "$OUT" \
  --prompt_style repocoder --max_input_tokens 16000 --control_arms --checkpoint_every 50 \
  "$@"
