"""P25-C — allin 三正例的 pill 精确定位（红equity + 蓝OUTS）+ 每座白色卡面。"""
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


def pills(arr, x0, y0, x1, y1, tag):
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    red = (r > 130) & (g < 70) & (b < 70)
    blue = (b > 100) & (b > r + 40) & (b > g + 30)
    white = (l > 190) & (r > 180) & (g > 180) & (b > 180)
    print(f"\n== {tag} ==")
    for nm, m in (("red", red), ("blue", blue), ("white", white)):
        n = int(m.sum())
        if n < 10:
            print(f"  {nm}: n={n}")
            continue
        ys, xs = np.where(m)
        # 行密度 → 主块
        rowd = m.sum(axis=1)
        rows = np.where(rowd >= max(6, rowd.max() * 0.3))[0]
        if len(rows):
            print(f"  {nm}: n={n} bbox=({x0+xs.min()},{y0+ys.min()},"
                  f"{xs.max()-xs.min()+1},{ys.max()-ys.min()+1}) "
                  f"dense_rows=({y0+rows.min()},{y0+rows.max()})")


a3 = load("POT_CHANGED_102_1787992505794_full.png")
pills(a3, 780, 30, 1120, 330, "f03 S2")
a4 = load("POT_CHANGED_13_1787992279195_full.png")
pills(a4, 100, 150, 480, 480, "f04 S1")
pills(a4, 780, 30, 1120, 330, "f04 S2")
