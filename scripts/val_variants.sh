#!/usr/bin/env bash
# Run pipeline variants on the 300-record validation set against an already-running model server
# and score each one. Use only to choose between variants, not to predict the leaderboard.
#   bash scripts/val_variants.sh            # all variants
#   bash scripts/val_variants.sh verify_off # one variant
set -uo pipefail
source "${VENV:-.venv}/bin/activate"
mkdir -p outputs/val logs
COMMON="--set input_file=val_input.csv"

run_variant() {  # name, [env prefix handled by caller], pipeline --set args...
  local name=$1; shift
  local out="outputs/val/$name"
  echo "=== $name  $(date -u +%T)"
  local t0=$(date +%s)
  python -m dischem.pipeline run --config config/default.yaml $COMMON \
      --set output_dir=$out --set cache_dir=cache/val_$name "$@" > logs/val_$name.log 2>&1
  local mins=$(python3 -c "print(round(($(date +%s)-$t0)/60,1))")
  python -m dischem.evaluate score --pred $out/submission.csv --ref data/val_reference.csv > $out/score.json
  python3 - "$name" "$mins" "$out" <<PY
import json,sys
n,m,o=sys.argv[1:]
s=json.load(open(f"{o}/score.json")); r=json.load(open(f"{o}/run_report.json"))
f=" ".join(f"{k[:6]}={v:.2f}" for k,v in s["fields"].items())
print(f"RESULT {n:14} overall={s['overall']:.4f} minutes={m} failed_calls={r['llm_stats']['failed']} | {f}")
PY
}

declare -A V=(
  [baseline]="" 
  [few_weak_off]="--set compose.few_shot_weak=0"
  [verify_off]="--set verify.enabled=false"
  [img1280]="--set images.max_side_px=1280"
  [img2]="--set images.max_per_record=2"
  [conc12]="--set llm.concurrency=12"
)
ORDER=(baseline few_weak_off verify_off img1280 img2 conc12)
[ $# -gt 0 ] && ORDER=("$@")
for n in "${ORDER[@]}"; do run_variant "$n" ${V[$n]}; done | tee -a logs/val_results.txt
