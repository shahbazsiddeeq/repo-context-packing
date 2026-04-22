#!/usr/bin/env bash
# Monitors both CrossCodeEval runs and auto-restarts them if they crash.
# Resumes from checkpoint automatically (the Python script handles this).
# Runs until both jobs have a summary file (success) or hit MAX_RETRIES.
#
# Usage: bash scripts/autowatch.sh
# Logs:  outputs/logs/autowatch.log

set -uo pipefail

REPO="/home/shahbaz/repo-context-packing"
PYTHON="$REPO/.venv-linux/bin/python"
SCRIPT="$REPO/scripts/run_crosscodeeval_official_prompts.py"
LOG_DIR="$REPO/outputs/logs"
MAX_RETRIES=10
POLL=60   # seconds between checks

log() { echo "[autowatch] $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG_DIR/autowatch.log"; }

# ── job definitions ────────────────────────────────────────────────────────────
# Each job: name, screen_name, gpu, output_dir, summary_file, extra_args
declare -A JOB_SCREEN=( [bm25]="cceval_bm25v2"   [prompt]="cceval_prompt"  )
declare -A JOB_GPU=(    [bm25]="1"                [prompt]="0"              )
declare -A JOB_OUTDIR=( [bm25]="outputs/cceval_python_hf_bm25_full"
                        [prompt]="outputs/cceval_python_hf_promptonly_full" )
declare -A JOB_CTX=(    [bm25]="auto"             [prompt]="never"          )
# prompt-only: limit to 4096 tokens to avoid eager-attention OOM on L4 (23.5 GB GPU).
# bm25: use default 16000 tokens; OOM on long examples is handled inside the script
#       via the oom-fallback loop (halves budget until it fits, min 512 tokens).
declare -A JOB_EXTRA=(  [bm25]=""
                        [prompt]="--max_input_tokens 4096"                 )
declare -A JOB_LOG=(    [bm25]="$LOG_DIR/cceval_bm25_full_v2.log"
                        [prompt]="$LOG_DIR/cceval_promptonly_full.log"       )
declare -A JOB_RETRIES=( [bm25]=0                 [prompt]=0                )
declare -A JOB_DONE=(    [bm25]=0                 [prompt]=0                )

summary_path() {
    echo "$REPO/${JOB_OUTDIR[$1]}/crosscodeeval_official_prompts_summary.txt"
}

screen_pid() {
    # Returns the PID of the Python process in the named screen, or empty string
    local sname="$1"
    # Get the pts/tty of the screen window, then find the python process on it
    local pid
    pid=$(pgrep -f "run_crosscodeeval_official_prompts.py.*${JOB_CTX[${2}]}" 2>/dev/null | head -1)
    echo "$pid"
}

is_screen_alive() {
    screen -ls 2>/dev/null | grep -q "${JOB_SCREEN[$1]}"
}

launch_job() {
    local job="$1"
    local sname="${JOB_SCREEN[$job]}"
    local gpu="${JOB_GPU[$job]}"
    local outdir="$REPO/${JOB_OUTDIR[$job]}"
    local ctx="${JOB_CTX[$job]}"
    local logfile="${JOB_LOG[$job]}"

    # Kill stale screen if exists
    screen -S "$sname" -X quit 2>/dev/null || true
    sleep 1

    local extra="${JOB_EXTRA[$job]}"
    log "Launching $job (GPU $gpu, context=$ctx $extra) → $outdir"
    screen -dmS "$sname" bash -c "
        cd $REPO
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u $SCRIPT \\
            --input cceval/data/python/line_completion_rg1_bm25.jsonl \\
            --output_dir $outdir \\
            --model_name bigcode/starcoder2-7b \\
            --max_rows all \\
            --use_crossfile_context $ctx \\
            --checkpoint_every 50 \\
            $extra \\
            2>&1 | tee -a $logfile
        echo '[${job}] process exited at \$(date)' >> $logfile
    "
}

# ── initial launch if not already running ─────────────────────────────────────
mkdir -p "$LOG_DIR"
log "=== autowatch started ==="

for job in bm25 prompt; do
    if [ -f "$(summary_path $job)" ]; then
        log "$job already has summary — marking done"
        JOB_DONE[$job]=1
    elif ! is_screen_alive "$job"; then
        log "$job screen not found — launching fresh"
        launch_job "$job"
    else
        log "$job screen already running — monitoring"
    fi
done

# ── main monitor loop ──────────────────────────────────────────────────────────
while true; do
    sleep "$POLL"

    all_done=1
    for job in bm25 prompt; do
        [ "${JOB_DONE[$job]}" -eq 1 ] && continue
        all_done=0

        if [ -f "$(summary_path $job)" ]; then
            log "$job COMPLETE — summary found"
            JOB_DONE[$job]=1
            log "=== $job SUMMARY ==="
            cat "$(summary_path $job)" | tee -a "$LOG_DIR/autowatch.log"
            continue
        fi

        if ! is_screen_alive "$job"; then
            retries="${JOB_RETRIES[$job]}"
            if [ "$retries" -ge "$MAX_RETRIES" ]; then
                log "ERROR: $job hit MAX_RETRIES ($MAX_RETRIES) — giving up"
                JOB_DONE[$job]=1  # stop trying
                continue
            fi
            JOB_RETRIES[$job]=$(( retries + 1 ))
            log "CRASH DETECTED: $job (retry ${JOB_RETRIES[$job]}/$MAX_RETRIES) — restarting from checkpoint"
            launch_job "$job"
        else
            # Still running — log progress from log file
            last=$(tail -1 "${JOB_LOG[$job]}" 2>/dev/null || echo "(no log yet)")
            log "$job alive | last line: $last"
        fi
    done

    if [ "$all_done" -eq 1 ]; then
        log "=== ALL JOBS COMPLETE — autowatch exiting ==="
        exit 0
    fi
done
