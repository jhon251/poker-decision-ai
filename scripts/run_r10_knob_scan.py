"""R10 KNOB — facing 修复后剥削旋钮再复扫驱动（分段跑 + 断点续跑）。

矩阵（任务书冻结）：{fish, freqbot} × {0.05, 0.30} × 6,000 手，每格 3 段
（g0/g1/g2 各 2,000 手；--hand-offset 推导与 M4 复扫逐位一致 → 同 deal_seed，
旧-vs-新可比性保留）。分段 ≤2,000 手（外杀窗口 ~45 min，段越小越安全），
每段独立落盘，死段只补该段；全部工件镜像到 docs/m4_gate/knob_r10/。

纪律（ENVIRONMENT）：单负载原则 —— 一次只跑本驱动（4 worker），不与
pytest/其他 bench 并发；单段 subprocess timeout=2400s（4 worker 实测
~18-24 min/段，超时即弃段由下次调用重跑）。

判定（summarize 脚本，本脚本只负责跑完落盘）：
  主判据 = 行为分叉：预算格之间同 deal_seed records 出现真实分叉
  （hero_actions 逐手比对，分叉手数 > 0）；辅以 bb/100 差值。
  fish 零分叉不再是闸门否决项（结构性原因 M3 已证：fish 下注仅 ~5%，
  偏离条件不触发）；freqbot 分叉 = "旋钮活了"证据。

用法（后台）：
  <repo>/.venv/Scripts/python.exe scripts/run_r10_knob_scan.py \
      > .tmp/r10_knob_driver.log 2>&1
可选 --opponents fish,freqbot（默认全部）。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "bench_r1"
COPY_DIR = ROOT / "docs" / "m4_gate" / "knob_r10"

SEGMENTS = [("g0", 0, 2000), ("g1", 2000, 2000), ("g2", 4000, 2000)]
ALL_ARMS = ["r10_knob_fish_b005", "r10_knob_fish_b030",
            "r10_knob_freqbot_b005", "r10_knob_freqbot_b030"]
WORKERS = 4  # R6 内存纪律 + R10 单负载原则：一次只有本驱动一个重负载


def seg_done(arm: str, seg: str) -> bool:
    return (OUT_DIR / f"{arm}.{seg}.json").is_file() \
        and (OUT_DIR / f"{arm}.{seg}.records.jsonl").is_file()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponents", default="fish,freqbot",
                    help="逗号分隔子集（默认全部）")
    args = ap.parse_args()
    opps = {s.strip() for s in args.opponents.split(",") if s.strip()}
    arms = [a for a in ALL_ARMS
            if any(a.startswith(f"r10_knob_{o}_") for o in opps)]

    plan = [(arm, seg, off, n) for arm in arms
            for seg, off, n in SEGMENTS if not seg_done(arm, seg)]
    total = len(plan)
    print(f"[driver] segments to run: {total} "
          f"({len(arms) * len(SEGMENTS) - total} already done)",
          flush=True)
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
            r = subprocess.run(cmd, cwd=str(ROOT), timeout=2400,
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
    done = sum(1 for arm in arms for seg, _, _ in SEGMENTS
               if seg_done(arm, seg))
    print(f"[driver] FINISHED {done}/{len(arms) * len(SEGMENTS)} segments "
          f"in {round((time.perf_counter() - t0) / 60, 1)}min", flush=True)
    return 0 if done == len(arms) * len(SEGMENTS) else 1


if __name__ == "__main__":
    sys.exit(main())
