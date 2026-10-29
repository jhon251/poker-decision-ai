"""P29-A — 干净样本精确 card bbox 测量（双卡分离：列投影 + 卡间空白 gap）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def card_boxes(fp, tag):
    arr = load(fp)
    roi = arr[990:1190, 790:1090]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    white = ((l > 150) & (r > 140) & (g > 140) & (b > 140)).astype(np.uint8)
    cold = white.sum(axis=0)
    runs, run = [], None
    for i, v in enumerate(cold):
        if v >= 20 and run is None:
            run = [i, i]
        elif v >= 20 and run is not None:
            run[1] = i
        elif run is not None:
            runs.append(run); run = None
    if run:
        runs.append(run)
    # gap: 两卡间白密度的谷（>=4px wide, <10 密度）
    if len(runs) < 2:
        print(f"[{tag}] runs={len(runs)}")
        return None
    # 直接用 gap 分析：找冷谷
    gaps = []
    for i in range(len(runs) - 1):
        seg = cold[runs[i][1]:runs[i + 1][0]]
        gaps.append((runs[i][1], runs[i + 1][0], int(seg.min())))
    best = max(gaps, key=lambda g: (g[2], g[1] - g[0]))
    cx0 = 790 + runs[0][0]
    bx0 = 790 + best[0] + 1
    bx1 = 790 + best[1]
    cx1 = 790 + runs[-1][1]
    # 行
    rowd = white.sum(axis=1)
    rows = np.where(rowd >= 40)[0]
    y0, y1 = 990 + rows.min(), 990 + rows.max()
    print(f"[{tag}] card1=({cx0},{y0})-({bx0},{y1}) "
          f"w={bx0-cx0+1} card2=({bx1},{y0})-({cx1},{y1}) "
          f"w={cx1-bx1+1} gap={bx1-bx0-1} h={y1-y0+1} "
          f"total=({cx0},{y0},{cx1},{y1})")


for fp, tag in [
        ("POT_CHANGED_110_1787992526057_full.png", "pc110 6c3h"),
        ("POT_CHANGED_16_1787995303501_full.png", "pc16b Kc3h"),
        ("POT_CHANGED_143_1787992603846_full.png", "pc143 6c3h"),
        ("POT_CHANGED_102_1787992505794_full.png", "pc102 9c3d(overlay)"),
        ("SCENE_CHANGED_74_1787992438991_full.png", "sc74 9c3d(overlay)")]:
    card_boxes(fp, tag)
