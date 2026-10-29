"""P25-B.3 — 定位 f03 S1 tag（dim 场景）→ 确认 S1 tag 固定位置。"""
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


def teal_mask(arr):
    r = arr[..., 0].astype(np.int16); g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    # teal: g > 1.5r, |g-b| <= 14, 有内容（排除纯黑/纯白）
    return ((g > r * 1.5) & (np.abs(g - b) <= 14) & (l > 6) & (l < 110))


arr = load("POT_CHANGED_102_1787992505794_full.png")
m = teal_mask(arr)
# S1 anchor (255,310)：S4 的 tag 相对锚点 off=(-117,+23)；
# 试探 S1 区域 x 100-330, y 250-420
sub = m[250:430, 100:340]
ys, xs = np.where(sub)
print("f03 S1 teal in (100,250)-(340,430): px=", len(xs))
if len(xs):
    print("bbox=", (100 + xs.min(), 250 + ys.min(), xs.max() - xs.min() + 1,
                    ys.max() - ys.min() + 1))
    # 逐列密度找精确块
    cold = sub.sum(axis=0)
    cols = np.where(cold >= 15)[0]
    rowd = sub.sum(axis=1)
    rows = np.where(rowd >= 15)[0]
    print("dense cols=", (100 + cols.min(), 100 + cols.max()),
          "dense rows=", (250 + rows.min(), 250 + rows.max()))
