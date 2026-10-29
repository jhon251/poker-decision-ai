"""P25-B.3 — tag 精确颜色/几何：背景色、暗字色、tag bbox、相对锚点偏移。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def load(frame):
    w, h, rgba = decode_png((EV / frame).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def stats(arr, x0, y0, x1, y1, tag):
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    # 聚类：暗字 (l<60) vs 中灰/teal bg (60<=l<130) vs 亮 (l>=130)
    dark = l < 60
    mid = (l >= 60) & (l < 130)
    bright = l >= 130
    print(f"[{tag}] dark={dark.sum()} mid={mid.sum()} bright={bright.sum()}")
    for name, m in (("dark", dark), ("mid", mid), ("bright", bright)):
        if m.sum() < 30:
            continue
        ys, xs = np.where(m)
        print(f"    {name}: bbox=({x0+xs.min()},{y0+ys.min()},"
              f"{xs.max()-xs.min()+1},{ys.max()-ys.min()+1}) "
              f"rgb_mean=({r[m].mean():.0f},{g[m].mean():.0f},"
              f"{b[m].mean():.0f})")


arr = load("BUTTON_CHANGED_167_1787992682749_full.png")
stats(arr, 1520, 880, 1650, 940, "f01_S5_tagzone")
arr6 = load("BUTTON_CHANGED_167_1787992682749_full.png")
stats(arr6, 800, 1130, 960, 1210, "f01_S6_tagzone")
