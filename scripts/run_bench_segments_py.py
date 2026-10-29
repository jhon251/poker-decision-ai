"""SOLVER-2 — bench 分段驱动（detached 进程用；断点续跑）。

依序跑完 solver2_on_vs_pivot（13×4000）与 solver2_on_vs_frequency_bot
（18×3000）；已存在的段工件跳过。每段一个独立 bench_b1.py 子进程
（<20 分钟，外杀长跑规避）；本驱动本身是 detached 进程（不经 bash 会话）。
"""
from __future__ import annotations

import glob
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

PLAN = [
    ("pivot", 13, 4000),
    ("frequency_bot", 18, 3000),
]


def seg_done(matchup: str, i: int) -> bool:
    s = f"s{i:02d}"
    return (ROOT / "docs" / "bench_r1" /
            f"solver2_on_vs_{matchup}.{s}.json").is_file()


def main() -> int:
    for matchup, n_seg, seg_hands in PLAN:
        for i in range(n_seg):
            if seg_done(matchup, i):
                print(f"[skip] {matchup} s{i:02d} exists", flush=True)
                continue
            s = f"s{i:02d}"
            off = i * seg_hands
            print(f"=== {matchup} {s} offset={off} hands={seg_hands} "
                  f"{time.strftime('%H:%M:%S')} ===", flush=True)
            log_p = ROOT / ".tmp" / f"bench_{matchup}_{s}.log"
            with open(log_p, "w", encoding="utf-8") as lf:
                rc = subprocess.call(
                    [PY, "scripts/bench_b1.py",
                     "--config", "configs/bench/bench_r1.yaml",
                     "--matchup", f"solver2_on_vs_{matchup}",
                     "--hands", str(seg_hands), "--workers", "8",
                     "--hand-offset", str(off), "--out-suffix", s],
                    stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT))
            print(f"EXIT={rc} {matchup} {s} {time.strftime('%H:%M:%S')}",
                  flush=True)
    print("ALL BENCH SEGMENTS DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
