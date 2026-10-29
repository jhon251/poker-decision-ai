"""M3 — 剥削预算旋钮扫描驱动（分段跑 + 断点续跑；ENVIRONMENT 协议）。

9 格矩阵（3 预算 × 3 对手）× 每格 10,000 手，每格拆 3 段
（k00=4000 / k01=4000 / k02=2000，bench_b1 --hand-offset 推导与长跑一致）。
本驱动**顺序**跑全部段（一次只有一个重负载——ENVIRONMENT 第 2 条），
已存在的段工件直接跳过（本机会外杀长跑 python → 重跑即续）。

用法（后台）：
  <repo>/.venv/Scripts/python.exe scripts/run_m3_knob_scan.py \
      > .tmp/knob_scan_driver.log 2>&1
之后：scripts/summarize_m3_knob_scan.py 汇总矩阵 + 推荐表。
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "bench_r1"

SEGMENTS = [("k00", 0, 4000), ("k01", 4000, 4000), ("k02", 8000, 2000)]
ARMS = ["m3_knob_freqbot_b005", "m3_knob_freqbot_b015", "m3_knob_freqbot_b030",
        "m3_knob_fish_b005", "m3_knob_fish_b015", "m3_knob_fish_b030",
        "m3_knob_reg_b005", "m3_knob_reg_b015", "m3_knob_reg_b030"]
WORKERS = 8


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
            r = subprocess.run(cmd, cwd=str(ROOT), timeout=3600,
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
    done = sum(1 for arm in ARMS for seg, _, _ in SEGMENTS
               if seg_done(arm, seg))
    print(f"[driver] FINISHED {done}/{len(ARMS) * len(SEGMENTS)} segments "
          f"in {round((time.perf_counter() - t0) / 60, 1)}min", flush=True)
    return 0 if done == len(ARMS) * len(SEGMENTS) else 1


if __name__ == "__main__":
    sys.exit(main())
