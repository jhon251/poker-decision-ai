"""P29-B — 精确 tight bbox：card1/card2 的 rank 与 suit（近角窗内自动收紧）。"""
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


def tight_ink(arr, x0, y0, x1, y1, tag):
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    ink = l < 120
    ys, xs = np.where(ink)
    if len(xs) < 12:
        print(f"[{tag}] ink<12 n={len(xs)}")
        return
    bb = (x0 + int(xs.min()), y0 + int(ys.min()),
          int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    red = (r > 110) & (r > g + 30) & (r > b + 30)
    print(f"[{tag}] tight=({bb[0]},{bb[1]},{bb[2]},{bb[3]}) "
          f"n={len(xs)} red_frac={float(red[ink].mean()):.2f}")


# 大窗口（涵盖旋转偏移）：
# card1 角标窗：x 813-880, y 1011-1105（rank 上半 / suit 下半）
# card2 角标窗：x 830-900, y 1005-1105
for fp, c1, c2 in [
        ("8bf82f134651db0add626be0304a2ff1.png", "5c", "3c"),
        ("POT_CHANGED_110_1787992526057_full.png", "6c", "3h"),
        ("POT_CHANGED_102_1787992505794_full.png", "9c", "3d")]:
    arr = load(fp)
    print(f"--- {fp[:28]} GT={c1}{c2} ---")
    # card1 rank zone (上 55px): (813,1011,880,1066)
    tight_ink(arr, 813, 1011, 880, 1066, "c1_rank_zone")
    # card1 suit zone: (813,1066,880,1120)
    tight_ink(arr, 813, 1066, 880, 1120, "c1_suit_zone")
    # card2 rank zone: (828,1005,900,1065)
    tight_ink(arr, 828, 1005, 900, 1065, "c2_rank_zone")
    # card2 suit zone: (828,1065,900,1125)
    tight_ink(arr, 828, 1065, 900, 1125, "c2_suit_zone")
