"""P13-Action5 — 最终证据分析：按钮定位 + 前后状态链 + 四类统计。

对 27 个 occurrence（用户标注）：按钮颜色/bbox 验证（红=盖牌、青=跟注/过牌、
橙=加注）+ 冻结观察者前后状态（pot/stack/badge/street）。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    PotObserver, StatusObserver, StackObserver)

SINCE = 134324658401327108
# 用户标注（occurrence -> 动作；? = 未标注）
ANNOT = {2: "fold", 3: "check", 4: "call", 6: "fold", 8: "fold", 9: "fold",
         10: "raise_pre", 11: "call", 12: "raise", 13: "allin", 14: "fold",
         15: "fold", 16: "fold", 17: "fold", 19: "call", 20: "fold",
         21: "fold", 22: "call", 23: "fold", 25: "fold"}


def load(p: Path) -> np.ndarray:
    w, h, rgba = decode_png(p.read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def band_sig(arr: np.ndarray) -> dict:
    """按钮带 (x1050-1876, y1200-1400) 三色像素量 + 白字量。"""
    x0, y0, x1, y1 = 1050, 1200, 1876, 1400
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    lum = r * 0.299 + g * 0.587 + b * 0.114
    R = (r > g + 50) & (r > b + 50) & (r > 110)
    O = (r > 140) & (g > 80) & (g < 190) & (b < 100)
    T = (g > r + 15) & (g >= b - 12) & (g > 50) & (g <= 135)
    W = (lum > 150) & (np.abs(r - g) < 40) & (np.abs(g - b) < 40)
    return {
        "red": int(R[y0:y1, x0:x1].sum()),
        "orange": int(O[y0:y1, x0:x1].sum()),
        "teal": int(T[y0:y1, x0:x1].sum()),
        "white": int(W[y0:y1, x0:x1].sum()),
    }


def main() -> int:
    since_s = SINCE / 1e7 - 11644473600
    new = [p for p in sorted(Path("data/vision2/coinpoker_evidence")
                             .glob("*_full.png"))
           if p.stat().st_mtime >= since_s]

    def ts(p):
        return int(re.search(r"_(\d+)_full$", p.stem).group(1)) / 1000.0
    new.sort(key=ts)
    # occurrence 帧聚类（红按钮 >12000）
    occ = []
    cur = None
    for i, p in enumerate(new):
        arr = load(p)
        r = arr[..., 0].astype(np.int16)
        g = arr[..., 1].astype(np.int16)
        b = arr[..., 2].astype(np.int16)
        m = ((r[1150:1400, :] > g[1150:1400, :] + 50)
             & (r[1150:1400, :] > b[1150:1400, :] + 50)
             & (r[1150:1400, :] > 110))
        if int(m.sum()) > 12000:
            if cur is None:
                cur = [i, i]
            elif ts(new[i]) - ts(new[cur[1]]) < 20:
                cur[1] = i
            else:
                occ.append(cur)
                cur = [i, i]
    if cur:
        occ.append(cur)
    pot_obs = PotObserver()
    status_obs = StatusObserver()
    stack_obs = StackObserver()
    report = []
    for n, (s, e) in enumerate(occ, 1):
        act = ANNOT.get(n, "?")
        # 按钮带峰值帧签名
        sigs = [band_sig(load(new[i])) for i in range(s, e + 1)]
        peak = max(sigs, key=lambda d: d["red"] + d["orange"])
        # 前后状态（该 occurrence 前后各 1-2 帧的冻结观察者）
        def states(flist):
            outs = []
            for p in flist:
                arr = load(p)
                w, h = arr.shape[:2]
                pot = pot_obs.observe(arr, w, h)
                status = status_obs.observe(arr, w, h)
                stack = stack_obs.observe(arr, w, h)
                outs.append({
                    "pot": pot.get("pot"),
                    "badges": {k: v.get("status") for k, v in
                               (status.get("seats") or {}).items()},
                })
            return outs
        before = states(new[max(0, s - 2):s])
        after = states(new[e + 1:min(len(new), e + 3)])
        report.append({
            "occ": n, "action": act,
            "t0": ts(new[s]), "t1": ts(new[e]),
            "frames": e - s + 1,
            "peak_band": peak,
            "before": before, "after": after,
        })
        print(f"occ{n:02d} {act:10s} 帧数={e-s+1:3d} "
              f"peak: red={peak['red']} orange={peak['orange']} "
              f"teal={peak['teal']} white={peak['white']} "
              f"pot前={[b['pot'] for b in before]} "
              f"pot后={[a['pot'] for a in after]}")
    Path("data/vision2/action5_evidence.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False, default=str),
        encoding="utf-8")
    from collections import Counter
    acts = Counter(a["action"] for a in report if a["action"] != "?")
    print("=== 独立实例统计 ===")
    for k, v in acts.most_common():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
