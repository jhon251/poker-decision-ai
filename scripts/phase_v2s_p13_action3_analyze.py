"""P13-Action3 — 手牌窗口聚类 + 状态信号（只读分析）。

- 新 950 帧：hero 红底手牌存在性 → 连续窗口 = 手牌实例（occurrence）；
- 每帧冻结 Pot/Status/Stack 观察者信号（状态变化证据）；
- 输出窗口表（帧区间 + 起止时间 + 状态变化摘要）供用户动作日志对齐。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    PotObserver, StatusObserver, StackObserver)

SINCE = 134324519952050945


def load(p: Path) -> np.ndarray:
    w, h, rgba = decode_png(p.read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def main() -> int:
    since_s = SINCE / 1e7 - 11644473600
    new = [p for p in sorted(Path("data/vision2/coinpoker_evidence")
                             .glob("*_full.png"))
           if p.stat().st_mtime >= since_s]
    pot_obs = PotObserver()
    status_obs = StatusObserver()
    stack_obs = StackObserver()
    rows = []
    for p in new:
        arr = load(p)
        w, h = arr.shape[:2]
        r = arr[..., 0].astype(np.int16)
        g = arr[..., 1].astype(np.int16)
        b = arr[..., 2].astype(np.int16)
        hero_red = int(((r[745:915, 163:407] > g[745:915, 163:407] + 50)
                        & (r[745:915, 163:407] > b[745:915, 163:407] + 50)
                        & (r[745:915, 163:407] > 110)).sum())
        pot = pot_obs.observe(arr, w, h)
        status = status_obs.observe(arr, w, h)
        stack = stack_obs.observe(arr, w, h)
        import re as _re
        ts = int(_re.search(r"_(\d+)_full$", p.stem).group(1)) / 1000.0
        rows.append({"stem": p.stem, "ts": ts, "hero_red": hero_red,
                     "hero_in_hand": hero_red > 3000,
                     "pot": pot.get("pot"),
                     "status": {s: v.get("status") for s, v in
                                (status.get("seats") or {}).items()},
                     "stack_S4": (stack.get("seats") or {}).get("S4", {})
                     .get("stack")})
    rows.sort(key=lambda r: r["ts"])       # 按时间排序（非文件名）
    # 手牌窗口（连续 hero_in_hand 帧，≤3 帧间隙合并）
    windows = []
    start = None
    prev_i = None
    for i, rw in enumerate(rows):
        if rw["hero_in_hand"]:
            if start is None or (prev_i is not None and i - prev_i > 3):
                if start is not None:
                    windows.append((start, prev_i))
                start = i
            prev_i = i
    if start is not None:
        windows.append((start, prev_i))
    import datetime as dt
    print(f"总帧={len(rows)} 手牌窗口={len(windows)}")
    for s, e in windows:
        t0 = dt.datetime.fromtimestamp(rows[s]["ts"]).strftime("%H:%M:%S")
        t1 = dt.datetime.fromtimestamp(rows[e]["ts"]).strftime("%H:%M:%S")
        pots = [rw["pot"] for rw in rows[s:e + 1] if rw["pot"]]
        sts = {}
        st4 = set()
        for rw in rows[s:e + 1]:
            for k, v in rw["status"].items():
                sts[k] = v
            if rw["stack_S4"] is not None:
                st4.add(rw["stack_S4"])
        print(f"  win {t0}-{t1} ({e - s + 1}帧) "
              f"pots={pots[:4]} status={sts} S4stacks={sorted(st4)[:5]}")
    Path("data/vision2/p13_action3_frames.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
