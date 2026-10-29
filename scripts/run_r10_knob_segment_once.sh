#!/usr/bin/env bash
# R10 KNOB 单段调用（通用）：跑一个分段并镜像工件到 docs/m4_gate/knob_r10/。
# 用法：bash scripts/run_r10_knob_segment_once.sh <arm> <seg> <offset> [hands]
#   arm ∈ r10_knob_{fish,freqbot}_{b005,b030}；seg 任意标签（g*/k*）；hands 默认 2000
set -u
ROOT="/c/poer"
ARM="$1"; SEG="$2"; OFF="$3"; HANDS="${4:-2000}"
LOG="$ROOT/.tmp/r10_knob_driver.log"
echo "$(date +%F' '%T) [once] $ARM.$SEG off=$OFF n=$HANDS start" >> "$LOG"
cd "$ROOT"
"$ROOT/.venv/Scripts/python.exe" scripts/bench_b1.py \
  --config configs/bench/bench_r1.yaml \
  --matchup "$ARM" --hands "$HANDS" --workers 4 \
  --hand-offset "$OFF" --out-suffix "$SEG" >> "$LOG" 2>&1
rc=$?
if [ "$rc" -eq 0 ] \
   && [ -f "$ROOT/docs/bench_r1/$ARM.$SEG.json" ] \
   && [ -f "$ROOT/docs/bench_r1/$ARM.$SEG.records.jsonl" ]; then
  mkdir -p "$ROOT/docs/m4_gate/knob_r10"
  cp "$ROOT/docs/bench_r1/$ARM.$SEG.json" "$ROOT/docs/m4_gate/knob_r10/"
  cp "$ROOT/docs/bench_r1/$ARM.$SEG.records.jsonl" "$ROOT/docs/m4_gate/knob_r10/"
  echo "$(date +%F' '%T) [once] $ARM.$SEG done (mirrored)" >> "$LOG"
else
  echo "$(date +%F' '%T) [once] $ARM.$SEG FAILED rc=$rc" >> "$LOG"
fi
exit "$rc"
