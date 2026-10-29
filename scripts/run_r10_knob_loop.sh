#!/usr/bin/env bash
# R10 KNOB 残余分段自驱循环（bash 常驻；外杀目标是 python，bash 循环免疫——
# 若被杀原样重启即可，缺哪段补哪段）。分段=1000 手（k0..k5，offset 0..5000），
# fish_b005 追加 k0..k5：预算格对照必须同布局（bot rng 消费 worker 内
# 相对 hand_index，g 段 500 手/worker 与 k 段 250 手/worker 不可配对）
# —— g* 工件保留作旧-vs-新对照（old g2 同为 4w×500/worker，可比）。
set -u
ROOT="/c/poer"
LOG="$ROOT/.tmp/r10_knob_driver.log"
# 单实例锁（mkdir 原子；>30 min 视为陈旧自动接管——防僵尸循环重复跑段）
LOCKDIR="$ROOT/.tmp/r10_knob_loop.lock"
if mkdir "$LOCKDIR" 2>/dev/null; then
  echo $$ > "$LOCKDIR/pid"; 
else
  age=$(( $(date +%s) - $(stat -c %Y "$LOCKDIR" 2>/dev/null || echo 0) ))
  if [ "$age" -gt 1800 ]; then
    echo "$(date +%F' '%T) [loop] stale lock (age=${age}s) — taking over" >> "$LOG"
    rm -rf "$LOCKDIR"; mkdir "$LOCKDIR"; echo $$ > "$LOCKDIR/pid"
  else
    echo "$(date +%F' '%T) [loop] another loop instance holds the lock (age=${age}s) — exit" >> "$LOG"
    exit 0
  fi
fi
trap 'rm -rf "$LOCKDIR"' EXIT
declare -a ARMS=(r10_knob_fish_b030 r10_knob_freqbot_b005 r10_knob_freqbot_b030 r10_knob_fish_b005)
echo "$(date +%F' '%T) [loop] knob residual loop start" >> "$LOG"
while :; do
  all_done=1
  for ARM in "${ARMS[@]}"; do
    for K in 0 1 2 3 4 5; do
      OFF=$((K * 1000))
      if [ -f "$ROOT/docs/bench_r1/$ARM.k$K.json" ] \
         && [ -f "$ROOT/docs/bench_r1/$ARM.k$K.records.jsonl" ]; then
        continue
      fi
      all_done=0
      bash "$ROOT/scripts/run_r10_knob_segment_once.sh" "$ARM" "k$K" "$OFF" 1000
      rc=$?
      echo "$(date +%F' '%T) [loop] $ARM.k$K rc=$rc" >> "$LOG"
      if [ "$rc" -ne 0 ]; then sleep 90; fi
    done
  done
  if [ "$all_done" -eq 1 ]; then
    echo "$(date +%F' '%T) [loop] all residual segments present — exit" >> "$LOG"
    break
  fi
done
