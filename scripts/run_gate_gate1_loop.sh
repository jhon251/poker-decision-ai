#!/usr/bin/env bash
# R10 GATE-1 战役自驱循环：反复调用单段执行器直到 50/50 段落盘。
# 理由：本机外杀周期性清扫高 CPU python（实测 23-48min 波动窗口）；bash 循环
# 不在清扫目标内（20:02 清扫实证：低 CPU python 存活）。每个分段 python 子进程
# 只活 ~25-40 min；被清扫杀死的段由循环自动重试（执行器缺段扫描 + 锁防重叠）。
# 被杀后原样重启本循环即可：bash scripts/run_gate_gate1_loop.sh
set -u
ROOT="/c/poer"
LOG="$ROOT/.tmp/gate_gate1_loop.log"
# 单实例锁（mkdir 原子；>50 min 视为陈旧自动接管——防僵尸循环重复跑段；
# 分段执行器另有文件锁防段内重叠）
LOCKDIR="$ROOT/.tmp/gate_gate1_loop.lock"
if mkdir "$LOCKDIR" 2>/dev/null; then
  echo $$ > "$LOCKDIR/pid"
else
  age=$(( $(date +%s) - $(stat -c %Y "$LOCKDIR" 2>/dev/null || echo 0) ))
  if [ "$age" -gt 3000 ]; then
    echo "$(date +%F' '%T) [loop] stale lock (age=${age}s) — taking over" >> "$LOG"
    rm -rf "$LOCKDIR"; mkdir "$LOCKDIR"; echo $$ > "$LOCKDIR/pid"
  else
    echo "$(date +%F' '%T) [loop] another loop instance holds the lock (age=${age}s) — exit" >> "$LOG"
    exit 0
  fi
fi
trap 'rm -rf "$LOCKDIR"' EXIT
PY="$ROOT/.venv/Scripts/python.exe"
TOTAL=50
echo "$(date +%F' '%T) gate_gate1 loop start" >> "$LOG"
while :; do
  done_n=$(ls "$ROOT/docs/m4_gate/gate_gate1"/s*.records.jsonl 2>/dev/null | wc -l)
  if [ "$done_n" -ge "$TOTAL" ]; then
    echo "$(date +%F' '%T) all $TOTAL segments present — loop exit" >> "$LOG"
    break
  fi
  "$PY" "$ROOT/scripts/run_m4_gate_segment_once.py" --campaign gate_gate1 \
    >> "$LOG" 2>&1
  rc=$?
  done_n=$(ls "$ROOT/docs/m4_gate/gate_gate1"/s*.records.jsonl 2>/dev/null | wc -l)
  echo "$(date +%F' '%T) segment rc=$rc done=$done_n/$TOTAL" >> "$LOG"
  if [ "$rc" -ne 0 ]; then sleep 120; fi
done
