#!/usr/bin/env bash
# Timed end-to-end run: start the model server, run the pipeline, stop the server.
# The clock starts before the model loads, so the reported time covers the whole inference run.
#   bash scripts/run_all.sh --limit 50      # benchmark: prints projected full-run minutes
#   bash scripts/run_all.sh                 # full run -> outputs/submission.csv
# Extra arguments are passed to the pipeline (e.g. --set llm.concurrency=12).
set -euo pipefail

VENV="${VENV:-.venv}"
# shellcheck disable=SC1091
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
mkdir -p outputs logs

START=$(date +%s)
echo "run started $(date -u +%FT%TZ)" | tee logs/run_timing.txt

bash scripts/serve_vllm.sh > logs/vllm.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

echo "waiting for the model server (log: logs/vllm.log) ..."
python -m dischem.pipeline run --config config/default.yaml "$@" 2>&1 | tee logs/pipeline.log
STATUS=${PIPESTATUS[0]}

END=$(date +%s)
MIN=$(python3 -c "print(round(($END-$START)/60,1))")
echo "run finished $(date -u +%FT%TZ)  total ${MIN} minutes (limit 180)" | tee -a logs/run_timing.txt
RATE="${POD_USD_PER_HOUR:-0.35}"
python3 -c "print(f'estimated cost at \${$RATE}/hour: \${($END-$START)/3600*$RATE:.2f} (limit \$20)')" | tee -a logs/run_timing.txt
exit "$STATUS"
