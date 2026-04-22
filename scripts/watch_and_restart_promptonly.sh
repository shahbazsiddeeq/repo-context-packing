#!/usr/bin/env bash
# Watches BM25 run (PID $1) and auto-starts prompt-only when it finishes.
# Usage: bash scripts/watch_and_restart_promptonly.sh <PID>

set -euo pipefail

BM25_PID="${1:-1183004}"
REPO="/home/shahbaz/repo-context-packing"
LOG_DIR="$REPO/outputs/logs"
SUMMARY="$REPO/outputs/cceval_python_hf_bm25_full/crosscodeeval_official_prompts_summary.txt"

echo "[watcher] Waiting for BM25 run PID $BM25_PID to finish..."

while kill -0 "$BM25_PID" 2>/dev/null; do
    sleep 30
    if [ -f "$SUMMARY" ]; then
        echo "[watcher] Summary file appeared — BM25 run complete."
        break
    fi
    echo "[watcher] $(date '+%H:%M:%S') BM25 still running..."
done

# Confirm result
if [ -f "$SUMMARY" ]; then
    echo "[watcher] BM25 summary:"
    cat "$SUMMARY"
else
    echo "[watcher] WARNING: BM25 PID $BM25_PID exited but no summary file found. May have crashed."
    echo "[watcher] Checking output dir..."
    ls -la "$REPO/outputs/cceval_python_hf_bm25_full/" || true
fi

echo ""
echo "[watcher] Starting prompt-only run on GPU 0 (Tesla P40)..."
sleep 5

cd "$REPO"
screen -dmS cceval_prompt bash -c "
    cd $REPO
    CUDA_VISIBLE_DEVICES=0 .venv-linux/bin/python -u scripts/run_crosscodeeval_official_prompts.py \\
        --input cceval/data/python/line_completion_rg1_bm25.jsonl \\
        --output_dir outputs/cceval_python_hf_promptonly_full \\
        --model_name bigcode/starcoder2-7b \\
        --max_rows all \\
        --use_crossfile_context never \\
        --checkpoint_every 50 \\
        2>&1 | tee -a $LOG_DIR/cceval_promptonly_full.log
    echo '[prompt-only] DONE at \$(date)'
"

echo "[watcher] Prompt-only launched in screen session 'cceval_prompt'."
echo "[watcher] Monitor with: screen -r cceval_prompt"
echo "[watcher] Log at: $LOG_DIR/cceval_promptonly_full.log"
