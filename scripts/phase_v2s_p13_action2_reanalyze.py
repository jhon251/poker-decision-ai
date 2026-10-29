"""P13-Action2 复分析：红底手牌 + 动作按钮行（红色按钮+白字+加注尺度）。

- hero 手牌 = 红底（red>3000 即持牌）；动作 UI = 手牌右方红按钮排；
- 重新扫描全部 844 帧：hero 红底手牌存在 + 动作按钮排存在 + 连续帧聚类 = occurrence。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

SINCE = 134324485474671084
HERO_CARDS = (163, 745, 407, 915)
# 动作按钮排区（手牌右方+深色背景上方：按钮 + 加注尺度）
ACTION_REGION = (405, 730, 950, 940)


def load(p: Path) -> np.ndarray:
    w, h, rgba = decode_png(p.read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def features(arr: np.ndarray) -> dict:
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    lum = r * 0.299 + g * 0.587 + b * 0.114
    x0, y0, x1, y1 = HERO_CARDS
    hero_red = int(((r > g + 50) & (r > b + 50) & (r > 110))
                   [y0:y1, x0:x1].sum())
    ax0, ay0, ax1, ay1 = ACTION_REGION
    ar = r[ay0:ay1, ax0:ax1]
    ag = g[ay0:ay1, ax0:ax1]
    ab = b[ay0:ay1, ax0:ax1]
    al = lum[ay0:ay1, ax0:ax1]
    red = int(((ar > ag + 50) & (ar > ab + 50) & (ar > 110)).sum())
    white = int(((al > 150) & (np.abs(ar - ag) < 40)
                 & (np.abs(ag - ab) < 40)).sum())
    green = int(((ag > ar + 40) & (ag > ab + 40) & (ag > 90)).sum())
    return {"hero_red": hero_red, "action_red": red, "action_white": white,
            "action_green": green}


def main() -> int:
    since_s = SINCE / 1e7 - 11644473600
    new = [p for p in sorted(Path("data/vision2/coinpoker_evidence")
                             .glob("*_full.png"))
           if p.stat().st_mtime >= since_s]
    rows = []
    for p in new:
        arr = load(p)
        f = features(arr)
        f["frame"] = p.stem
        rows.append(f)
    # hero 红底手牌 = red > 3000；动作 UI = action_red > 2500 或 action_white > 1500
    hero_frames = sum(1 for f in rows if f["hero_red"] > 3000)
    action_frames = sum(1 for f in rows
                        if f["action_red"] > 2500 or f["action_white"] > 1500)
    both = sum(1 for f in rows
               if f["hero_red"] > 3000
               and (f["action_red"] > 2500 or f["action_white"] > 1500))
    print(f"总帧={len(rows)} hero红底手牌帧={hero_frames} "
          f"动作UI帧={action_frames} 两者同时={both}")
    # 动作 UI 帧的连续聚类 = occurrences（按帧序）
    act_idx = [i for i, f in enumerate(rows)
               if f["action_red"] > 2500 or f["action_white"] > 1500]
    occ = []
    if act_idx:
        start = act_idx[0]
        prev = act_idx[0]
        for i in act_idx[1:]:
            if i - prev > 4:
                occ.append((start, prev))
                start = i
            prev = i
        occ.append((start, prev))
    print("动作 UI occurrences（连续帧簇）:", len(occ))
    for s, e in occ:
        frs = [rows[i] for i in range(s, e + 1)]
        print(f"  occ 帧 {rows[s]['frame'][:24]} ~ {rows[e]['frame'][:24]} "
              f"({e - s + 1} 帧) hero_red={max(f['hero_red'] for f in frs)} "
              f"action_red={max(f['action_red'] for f in frs)} "
              f"action_white={max(f['action_white'] for f in frs)}")
    Path("data/vision2/p13_action2_reanalysis.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
