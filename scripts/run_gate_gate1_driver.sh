#!/usr/bin/env bash
# R10 GATE-1 战役后台驱动循环：反复调用单段执行器直到 50/50 段落盘。
# 设计：bash 循环常驻（外杀观察对象是长 python 进程；每个分段 python 子进程
# 只活 ~25-33 min < ~45 min 外杀窗口）；执行器自带锁 + 缺段扫描 + 断点续跑，
# 本循环被杀后原样重启即可（缺哪段补哪段）。失败退避 120s 防热旋。
set -u
ROOT="/c/poer"
OUT="$ROOT/docs/m4_gate/gate_gate1"
LOG="$ROOT/.tmp/gate_gate1_loop.log"
PY="$ROOT/.venv/Scripts/python.exe"
TOTAL=50
echo "$(date +%F' '%T) gate_gate1 driver loop start" >> "$LOG"
while :; do
  done_n=$(ls "$OUT"/s*.records.jsonl 2>/dev/null | wc -l)
  if [ "$done_n" -ge "$TOTAL" ]; then
    echo "$(date +%F' '%T) all $TOTAL segments present — loop exit" >> "$LOG"
    break
  fi
  "$PY" "$ROOT/scripts/run_m4_gate_segment_once.py" --campaign gate_gate1 \
    >> "$LOG" 2>&1
  rc=$?
  done_n=$(ls "$OUT"/s*.records.jsonl 2>/dev/null | wc -l)
  echo "$(date +%F' '%T) segment rc=$rc done=$done_n/$TOTAL" >> "$LOG"
  if [ "$rc" -ne 0 ]; then
    sleep 120
  fi
done
