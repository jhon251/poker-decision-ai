"""M4 Phase A3 — 旋钮复扫驱动（Phase Gate；分段跑 + 断点续跑）。

矩阵（任务书冻结）：{fish, freqbot} × {0.05, 0.30} × 6,000 手，每格 3 段
（g0/g1/g2 各 2,000 手；--hand-offset 推导与长跑逐位一致）。
分段 ≤2,000 手：本机外杀窗口实测 ~45 分钟（8 worker），段越小越安全——
每段独立落盘，死段只补该段（R6 收尾协议）。

闸门条件（summarize 脚本判定，本脚本只负责跑完落盘）：
  (a) 预算格之间 records 真实分叉（同段 md5 不同 + 逐手 hero_actions
      分叉手数 > 0）；
  (b) 软对手（fish）上高预算 bb/100 显著高于低预算（配对逐手差
      95% CI 下界 > 0）。

用法（后台）：
  <repo>/.venv/Scripts/python.exe scripts/run_m4_knob_scan.py \
      > .tmp/m4_knob_driver.log 2>&1
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "bench_r1"
COPY_DIR = ROOT / "docs" / "m4_gate" / "knob"

SEGMENTS = [("g0", 0, 2000), ("g1", 2000, 2000), ("g2", 4000, 2000)]
ARMS = ["m4_knob_fish_b005", "m4_knob_fish_b030",
        "m4_knob_freqbot_b005", "m4_knob_freqbot_b030"]
WORKERS = 4  # R6 内存纪律：8 worker 曾致机器 OOM 重启，一律 4


def seg_done(arm: str, seg: str) -> bool:
    return (OUT_DIR / f"{arm}.{seg}.json").is_file() \
        and (OUT_DIR / f"{arm}.{seg}.records.jsonl").is_file()


def main() -> int:
    plan = [(arm, seg, off, n) for arm in ARMS
            for seg, off, n in SEGMENTS if not seg_done(arm, seg)]
    total = len(plan)
    print(f"[driver] segments to run: {total} "
          f"({len(ARMS) * len(SEGMENTS) - total} already done)", flush=True)
    t0 = time.perf_counter()
    for i, (arm, seg, off, n) in enumerate(plan, 1):
        cmd = [sys.executable, str(ROOT / "scripts" / "bench_b1.py"),
               "--config", "configs/bench/bench_r1.yaml",
               "--matchup", arm, "--hands", str(n), "--workers", str(WORKERS),
               "--hand-offset", str(off), "--out-suffix", seg]
        print(f"[driver] ({i}/{total}) {arm}.{seg} off={off} n={n} "
              f"elapsed={round((time.perf_counter() - t0) / 60, 1)}min",
              flush=True)
        t_seg = time.perf_counter()
        try:
            r = subprocess.run(cmd, cwd=str(ROOT), timeout=5400,
                               capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            tail = (r.stdout or "").strip().splitlines()[-3:]
            for ln in tail:
                print(f"[driver]   | {ln}", flush=True)
            if r.returncode != 0:
                print(f"[driver]   !! exit={r.returncode} "
                      f"stderr={(r.stderr or '')[-400:]}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"[driver]   !! timeout (segment abandoned; "
                  f"rerun driver to retry)", flush=True)
            continue
        except Exception as exc:                 # noqa: BLE001
            print(f"[driver]   !! {exc}", flush=True)
            continue
        print(f"[driver] ({i}/{total}) done in "
              f"{round((time.perf_counter() - t_seg) / 60, 1)}min", flush=True)
        if seg_done(arm, seg):
            COPY_DIR.mkdir(parents=True, exist_ok=True)
            for ext in (".json", ".records.jsonl"):
                src = OUT_DIR / f"{arm}.{seg}{ext}"
                if src.is_file():
                    (COPY_DIR / f"{arm}.{seg}{ext}").write_bytes(
                        src.read_bytes())
    done = sum(1 for arm in ARMS for seg, _, _ in SEGMENTS
               if seg_done(arm, seg))
    print(f"[driver] FINISHED {done}/{len(ARMS) * len(SEGMENTS)} segments "
          f"in {round((time.perf_counter() - t0) / 60, 1)}min", flush=True)
    return 0 if done == len(ARMS) * len(SEGMENTS) else 1


if __name__ == "__main__":
    sys.exit(main())
