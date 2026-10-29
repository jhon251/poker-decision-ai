"""P29-B — 每卡 rank/suit 角标精确布局测量（高倍放大目视）。

3 合法 GT：8bf8=5c3c / pc110=6c3h / f03=9c3d（overlay）。
卡几何（P29-A）：card1=(813,1011,121,178) card2=(837,1011,122,178)。
rank 区 (795,1010,80,65)、suit 区 (795,1050,85,80) 为总区——需按卡细分。
用目视 + 暗/红墨投影精确定位每张卡的 rank 角标 bbox 与 suit 角标 bbox。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p29b_crops")
OUT.mkdir(parents=True, exist_ok=True)

# 帧与预期（GT）
GT = [
    ("8bf82f134651db0add626be0304a2ff1.png", "5c", "3c", "NORMAL"),
    ("POT_CHANGED_110_1787992526057_full.png", "6c", "3h", "NORMAL"),
    ("POT_CHANGED_102_1787992505794_full.png", "9c", "3d", "DIM+overlay"),
]
# card bbox（每帧实测——卡1左缘 812-813，卡2 左缘 836-837；宽 ~121/122）
CARD1 = (813, 1011, 121, 178)
CARD2 = (837, 1011, 122, 178)


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def ink_bbox(arr, x0, y0, w, h, tag):
    """白卡面内 暗墨+红墨 的 tight bbox（角标字形）。"""
    roi = arr[y0:y0 + h, x0:x0 + w]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    # 白卡面外的暗背景排除：只取卡面内（l>140 为白面，字为暗/红）
    ink = (l < 120)
    ys, xs = np.where(ink)
    if len(xs) < 15:
        print(f"[{tag}] ink<15 n={len(xs)}")
        return None
    bb = (x0 + int(xs.min()), y0 + int(ys.min()),
          int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    # 红墨比例
    red = ((r > 110) & (r > g + 30) & (r > b + 30))
    print(f"[{tag}] ink bbox=({bb[0]},{bb[1]},{bb[2]},{bb[3]}) "
          f"n={len(xs)} red_frac={float(red[ink].mean()):.2f}")
    return bb


for fp, c1, c2, cond in GT:
    arr = load(fp)
    print(f"\n=== {fp[:32]} ({cond}) GT={c1}{c2} ===")
    # card1 rank/suit（卡左上角，卡面顶部 ~35%）
    ink_bbox(arr, CARD1[0], CARD1[1], CARD1[2], 60, f"{fp[:6]} c1_rank")
    ink_bbox(arr, CARD1[0], CARD1[1] + 55, CARD1[2], 70, f"{fp[:6]} c1_suit")
    # card2 rank/suit
    ink_bbox(arr, CARD2[0], CARD2[1], CARD2[2], 60, f"{fp[:6]} c2_rank")
    ink_bbox(arr, CARD2[0], CARD2[1] + 55, CARD2[2], 70, f"{fp[:6]} c2_suit")
