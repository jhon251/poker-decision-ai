"""P29-B — card2 角标真位置：行投影分带（rank 带 / suit 带）+ 每带 tight 列。"""
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


# card2 = (837,1011)。角标区：x 837-920, y 1011-1120（不含卡缘重叠）
fp = "8bf82f134651db0add626be0304a2ff1.png"
arr = load(fp)
roi = arr[1005:1130, 840:925]      # rel card2: x+3..+88, y-6..+119
r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
b = roi[..., 2].astype(np.int16)
l = r * 0.299 + g * 0.587 + b * 0.114
ink = l < 120
rowd = ink.sum(axis=1)
print("row profile (y rel card2 -6..119):")
for i, v in enumerate(rowd):
    if v > 0:
        pass
runs, run = [], None
for i, v in enumerate(rowd):
    if v >= 2 and run is None:
        run = [i, i]
    elif v >= 2 and run is not None:
        run[1] = i
    elif run is not None:
        runs.append(run); run = None
if run:
    runs.append(run)
merged = []
for rr in runs:
    if merged and rr[0] - merged[-1][1] <= 3:
        merged[-1][1] = rr[1]
    else:
        merged.append(rr)
print("bands(rel card2):", [(a - 6, b - 6, b - a + 1) for a, b in merged])
for a, b in merged:
    sub = ink[a:b + 1]
    xs = np.where(sub.sum(axis=0) > 0)[0]
    if len(xs):
        print(f"  band y{a-6}..{b-6}: x rel {xs.min()+3}..{xs.max()+3}")
