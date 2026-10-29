#!/usr/bin/env bash
# SOLVER-2 — bench 分段驱动（断点续跑：已有段工件跳过）。
# 用法：bash scripts/run_bench_solver2_segments.sh pivot|frequency_bot
set -u
cd /c/poer
PY=".venv/Scripts/python.exe"
MATCHUP="$1"
case "$MATCHUP" in
  pivot)         N_SEG=13; SEG_HANDS=4000;;   # 13×4000 = 52,000
  frequency_bot) N_SEG=18; SEG_HANDS=3000;;   # 18×3000 = 54,000
  *) echo "usage: $0 pivot|frequency_bot"; exit 2;;
esac
for i in $(seq 0 $((N_SEG-1))); do
  S=$(printf "s%02d" "$i")
  OFF=$((i * SEG_HANDS))
  OUT="docs/bench_r1/solver2_on_vs_${MATCHUP}.${S}.json"
  if [ -f "$OUT" ]; then
    echo "[skip] $S exists ($OUT)"
    continue
  fi
  echo "=== segment $S offset=$OFF hands=$SEG_HANDS $(date '+%H:%M:%S') ==="
  "$PY" scripts/bench_b1.py --config configs/bench/bench_r1.yaml \
    --matchup "solver2_on_vs_${MATCHUP}" --hands "$SEG_HANDS" --workers 8 \
    --hand-offset "$OFF" --out-suffix "$S" \
    > ".tmp/bench_${MATCHUP}_${S}.log" 2>&1
  echo "EXIT=$? segment $S $(date '+%H:%M:%S')"
  tail -1 ".tmp/bench_${MATCHUP}_${S}.log"
done
echo "ALL SEGMENTS DONE $MATCHUP"
