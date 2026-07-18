#!/usr/bin/env bash
# Autoresearch loop: alternate an agent iteration (proposes + implements one
# experiment in submission.py) with the deterministic harness (benchmarks it
# on Modal, keeps or reverts, appends to the ledger).
#
# Usage:
#   ./autoresearch.sh [iterations]     # default 5
#
# Environment:
#   AR_GPU                 GPU for run_modal.py (default B200)
#   AR_IMPROVE_THRESHOLD   relative geomean gain required to keep a change
#                          (default 0.02)
#   CLAUDE_ARGS            extra args for claude -p (e.g. "--model opus")
#
# Everything is logged under autoresearch/logs/. Stop anytime with Ctrl-C;
# state lives in autoresearch/ledger.jsonl + autoresearch/best/, so the next
# run resumes where this one left off.

set -uo pipefail
cd "$(dirname "$0")"

ITERATIONS="${1:-5}"
AR=autoresearch
mkdir -p "$AR/logs"

command -v claude-third-party-opus-4.8 >/dev/null || { echo "claude CLI not found in PATH" >&2; exit 1; }

# Baseline: measure the current submission.py once so every delta has a
# denominator. Only runs when no best-known snapshot exists yet.
if [ ! -f "$AR/best/best.json" ]; then
    echo "=== no baseline yet: measuring current submission.py ==="
    /home/ubuntu/.venvs/cholesky/bin/python "$AR/run_experiment.py" --baseline 2>&1 | tee "$AR/logs/baseline.log"
    [ -f "$AR/best/best.json" ] || { echo "baseline failed; see $AR/logs/baseline.log" >&2; exit 1; }
fi

consecutive_agent_failures=0

for i in $(seq 1 "$ITERATIONS"); do
    ts=$(date +%Y%m%d_%H%M%S)
    log="$AR/logs/iter_${ts}.log"
    echo ""
    echo "=== iteration $i/$ITERATIONS ($ts) ===" | tee "$log"

    rm -f "$AR/CANDIDATE.json"

    # Agent half: reads ledger + backlog, edits submission.py, writes
    # CANDIDATE.json. It has no Modal access; only file edits are auto-allowed.
    claude-third-party-opus-4.8 -p "$(cat "$AR/RESEARCH.md")" \
        --permission-mode acceptEdits \
        --allowed-tools "Read,Glob,Grep,Edit,Write,Bash(/home/ubuntu/.venvs/cholesky/bin/python -m py_compile:*),Bash(/home/ubuntu/.venvs/cholesky/bin/python -m py_compile:*),WebFetch(domain:github.com),WebFetch(domain:raw.githubusercontent.com)" \
        --max-turns 50 \
        ${CLAUDE_ARGS:-} \
        2>&1 | tee -a "$log"

    if [ ! -f "$AR/CANDIDATE.json" ]; then
        echo "agent wrote no CANDIDATE.json; restoring best submission" | tee -a "$log"
        cp "$AR/best/submission.py" submission.py
        consecutive_agent_failures=$((consecutive_agent_failures + 1))
        if [ "$consecutive_agent_failures" -ge 3 ]; then
            echo "3 consecutive agent failures; aborting" | tee -a "$log"
            break
        fi
        continue
    fi
    consecutive_agent_failures=0

    if /home/ubuntu/.venvs/cholesky/bin/python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('stop') else 1)" "$AR/CANDIDATE.json"; then
        echo "agent requested stop:" | tee -a "$log"
        /home/ubuntu/.venvs/cholesky/bin/python -m json.tool "$AR/CANDIDATE.json" | tee -a "$log"
        break
    fi

    # Deterministic half: compile gate, Modal correctness tests, Modal
    # benchmarks, keep-or-revert, ledger append.
    /home/ubuntu/.venvs/cholesky/bin/python "$AR/run_experiment.py" --candidate "$AR/CANDIDATE.json" 2>&1 | tee -a "$log"
done

echo ""
echo "=== ledger summary ==="
/home/ubuntu/.venvs/cholesky/bin/python "$AR/run_experiment.py" --summary