"""P15 — 1876 几何精确测量（只读）：board 槽位 / hero 卡 / pot / 座栈 / bet 胶囊。"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

SINCE5 = 134324658401327108
SINCE6 = 134324688462724951


def _ts(p):
    return int(re.search(r"_(\d+)_full$", p.stem).group(1)) / 1000.0


def load(p):
    w, h, rgba = decode_png(p.read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def _frames():
    out = []
    for since in (SINCE5, SINCE6):
        s = since / 1e7 - 11644473600
        out += [p for p in sorted(Path("data/vision2/coinpoker_evidence")
                                  .glob("*_full.png"))
                if p.stat().st_mtime >= s]
    out.sort(key=_ts)
    return out


def white_rects(arr, y0, y1, x0, x1, min_w=60, min_h=60, min_area=4000):
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    lum = r * 0.299 + g * 0.587 + b * 0.114
    W = ((lum > 150) & (np.abs(r - g) < 40) & (np.abs(g - b) < 40)
         )[y0:y1, x0:x1]
    rows = W.sum(axis=1)
    bands, run = [], None
    for i, v in enumerate(rows):
        if v > 50 and run is None:
            run = i
        elif v <= 50 and run is not None:
            bands.append((run, i))
            run = None
    if run is not None:
        bands.append((run, len(rows)))
    out = []
    for bs, be in bands:
        if be - bs < min_h:
            continue
        band = W[bs:be]
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
            if ce - cs < min_w:
                continue
            if band[:, cs:ce].sum() < min_area:
                continue
            out.append((x0 + cs, y0 + bs, ce - cs, be - bs))
    return out


def gold_rects(arr, y0, y1, x0, x1, min_w=25, min_h=20, min_area=300):
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    lum = r * 0.299 + g * 0.587 + b * 0.114
    G = ((r > 120) & (g > 85) & (r >= g) & (g >= b - 30) & (lum > 90)
         )[y0:y1, x0:x1]
    rows = G.sum(axis=1)
    bands, run = [], None
    for i, v in enumerate(rows):
        if v > 10 and run is None:
            run = i
        elif v <= 10 and run is not None:
            bands.append((run, i))
            run = None
    if run is not None:
        bands.append((run, len(rows)))
    out = []
    for bs, be in bands:
        if be - bs < min_h:
            continue
        band = G[bs:be]
        cols = band.sum(axis=0)
        runs, run = [], None
        for i, v in enumerate(cols):
            if v > 6 and run is None:
                run = i
            elif v <= 6 and run is not None:
                runs.append((run, i))
                run = None
        if run is not None:
            runs.append((run, len(cols)))
        for cs, ce in runs:
            if ce - cs < min_w:
                continue
            if band[:, cs:ce].sum() < min_area:
                continue
            out.append((x0 + cs, y0 + bs, ce - cs, be - bs))
    return out


def main() -> int:
    frames = _frames()
    print("1876 帧数:", len(frames))
    # 采集 board / hero / pot / stack 位置统计（多帧）
    from collections import Counter
    board_c = Counter()
    hero_c = Counter()
    pot_c = Counter()
    stack_c = Counter()
    n = 0
    for p in frames:
        arr = load(p)
        if arr.shape[:2] != (1400, 1876):
            continue
        n += 1
        # board 区 (520-1220, 540-740) 白卡
        for (x, y, w, h) in white_rects(arr, 540, 740, 520, 1220):
            board_c[(x // 10 * 10, w // 20 * 20, h // 20 * 20)] += 1
        # hero 卡 (700-1120, 990-1200)
        for (x, y, w, h) in white_rects(arr, 990, 1200, 700, 1120):
            hero_c[(x // 10 * 10, w // 20 * 20, h // 20 * 20)] += 1
        # pot 顶中 (700-1150, 80-180) 金文本
        for (x, y, w, h) in gold_rects(arr, 80, 180, 700, 1150):
            pot_c[(x // 20 * 20, y // 20 * 20)] += 1
        # 各座栈金文本（底部带 y 1150-1350 全宽）
        for (x, y, w, h) in gold_rects(arr, 1150, 1350, 0, 1876, min_w=20):
            stack_c[(x // 30 * 30, y // 20 * 20)] += 1
    print("board 白卡簇(采样 n=%d):" % n, board_c.most_common(12))
    print("hero 白卡簇:", hero_c.most_common(8))
    print("pot 金文本簇:", pot_c.most_common(8))
    print("stack 金文本簇:", stack_c.most_common(20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
