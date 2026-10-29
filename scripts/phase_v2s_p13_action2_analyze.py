"""P13-Action2 — 只读证据分析：hero 入局 / Action UI 定位 / 语义链。

- 仅分析本轮新采集帧（since FileTime）；
- 每帧：hero 卡区 (163-407,745-915) 内容 · 全帧饱和色按钮候选 ·
  冻结 Pot/Status/Stack 观察者状态信号；
- 输出：hero-in-hand 帧、候选聚类、按钮出现/消失与状态变化的时序。
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

SINCE_FILETIME = 134324485474671084
HERO_CARDS = (163, 745, 407, 915)      # hero 卡区（S4 左下，hero_card_layout_v1）


def load(p: Path) -> np.ndarray:
    w, h, rgba = decode_png(p.read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def hero_cards_state(arr: np.ndarray) -> str:
    x0, y0, x1, y1 = HERO_CARDS
    sub = arr[y0:y1, x0:x1]
    r = sub[..., 0].astype(np.int16)
    g = sub[..., 1].astype(np.int16)
    b = sub[..., 2].astype(np.int16)
    lum = r * 0.299 + g * 0.587 + b * 0.114
    white = int(((lum > 150) & (np.abs(r - g) < 40)
                 & (np.abs(g - b) < 40)).sum())
    red = int(((r > g + 50) & (r > b + 50) & (r > 110)).sum())
    if white > 8000 and red > 300:
        return "CARDS"
    if white > 8000:
        return "PANEL"
    return "EMPTY"


def candidates(arr: np.ndarray, y_min: int = 0, y_max: int = 1066) -> list:
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    lum = r * 0.299 + g * 0.587 + b * 0.114
    masks = {
        "red": (r > g + 50) & (r > b + 50) & (r > 110),
        "green": (g > r + 40) & (g > b + 40) & (g > 90),
        "blue": (b > r + 40) & (b > g + 40) & (b > 100),
        "yellow": (r > 150) & (g > 110) & (b < 90) & (np.abs(r - g) < 90),
    }
    out = []
    for color, mask in masks.items():
        m = mask[y_min:y_max]
        rows = m.sum(axis=1)
        bands, run = [], None
        for i, v in enumerate(rows):
            if v > 100 and run is None:
                run = i
            elif v <= 100 and run is not None:
                bands.append((run, i))
                run = None
        if run is not None:
            bands.append((run, len(rows)))
        for bs, be in bands:
            if not (30 <= be - bs <= 140):
                continue
            band = m[bs:be]
            cols = band.sum(axis=0)
            runs, run = [], None
            for i, v in enumerate(cols):
                if v > 30 and run is None:
                    run = i
                elif v <= 30 and run is not None:
                    runs.append((run, i))
                    run = None
            if run is not None:
                runs.append((run, len(cols)))
            for cs, ce in runs:
                if not (50 <= ce - cs <= 500):
                    continue
                sub = band[:, cs:ce]
                if sub.sum() < 3000:
                    continue
                # 白字量（按钮标签证据）
                wsub = lum[y_min + bs:y_min + be, cs:ce]
                rs = r[y_min + bs:y_min + be, cs:ce]
                gs = g[y_min + bs:y_min + be, cs:ce]
                bs2 = b[y_min + bs:y_min + be, cs:ce]
                white = int(((wsub > 150) & (np.abs(rs - gs) < 40)
                             & (np.abs(gs - bs2) < 40)).sum())
                out.append({"color": color,
                            "bbox": [int(cs), int(bs + y_min),
                                     int(ce - cs), int(be - bs)],
                            "area": int(sub.sum()), "white": white})
    return out


def main() -> int:
    since_s = SINCE_FILETIME / 1e7 - 11644473600
    fulls = sorted(Path("data/vision2/coinpoker_evidence").glob("*_full.png"))
    new = [p for p in fulls if p.stat().st_mtime >= since_s]
    print("new_frames:", len(new))
    pot_obs = PotObserver()
    status_obs = StatusObserver()
    stack_obs = StackObserver()
    per_frame = []
    for path in new:
        arr = load(path)
        w, h = arr.shape[:2]
        pot = pot_obs.observe(arr, w, h)
        status = status_obs.observe(arr, w, h)
        stack = stack_obs.observe(arr, w, h)
        fr = {
            "frame": path.stem, "hero": hero_cards_state(arr),
            "cands": candidates(arr),
            "pot": pot.get("pot"),
            "status": {s: v.get("status") for s, v in
                       (status.get("seats") or {}).items()},
            "stack_S4": (stack.get("seats") or {}).get("S4", {}).get("stack"),
            "stack_S5": (stack.get("seats") or {}).get("S5", {}).get("stack"),
        }
        per_frame.append(fr)
    Path("data/vision2/p13_action2_frames.json").write_text(
        json.dumps(per_frame, indent=1, ensure_ascii=False), encoding="utf-8")
    # 统计
    from collections import Counter
    hc = Counter(fr["hero"] for fr in per_frame)
    print("hero-region states:", dict(hc))
    with_cards = [fr for fr in per_frame if fr["hero"] == "CARDS"]
    print("frames with hero cards:", len(with_cards))
    # 按钮候选聚类（含 hero 卡的帧内）
    clusters = {}
    for fr in per_frame:
        for c in fr["cands"]:
            key = (c["color"], c["bbox"][1] // 30, c["bbox"][0] // 30)
            d = clusters.setdefault(key, {"n": 0, "frames": [], "s": c})
            d["n"] += 1
            if len(d["frames"]) < 6:
                d["frames"].append(fr["frame"])
    print("=== candidate clusters (top 25) ===")
    for d in sorted(clusters.values(), key=lambda d: -d["n"])[:25]:
        s = d["s"]
        print(f"n={d['n']:4d} {s['color']:7s} bbox={s['bbox']} "
              f"area={s['area']} white={s['white']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
