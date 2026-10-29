"""P29-B — 精确定位 card2 digit/suit（窗口 930-990+）。"""
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


arr = load("8bf82f134651db0add626be0304a2ff1.png")
# digit 带 y 1005-1060：列投影 x 920-1010
roi = arr[1005:1060, 920:1010]
l = roi[..., 0] * 0.299 + roi[..., 1] * 0.587 + roi[..., 2] * 0.114
ink = (l < 120).astype(np.uint8)
cold = ink.sum(axis=0)
bands = []
run = None
for i, v in enumerate(cold):
    if v >= 3 and run is None:
        run = [i, i]
    elif v >= 3 and run is not None:
        run[1] = i
    elif run is not None:
        bands.append(run); run = None
if run:
    bands.append(run)
print("digit x-bands (abs):", [(920 + a, 920 + b, b - a + 1) for a, b in bands])
# suit 带 y 1063-1125
roi2 = arr[1063:1125, 920:1010]
l2 = roi2[..., 0] * 0.299 + roi2[..., 1] * 0.587 + roi2[..., 2] * 0.114
ink2 = (l2 < 120).astype(np.uint8)
cold2 = ink2.sum(axis=0)
bands2 = []
run = None
for i, v in enumerate(cold2):
    if v >= 3 and run is None:
        run = [i, i]
    elif v >= 3 and run is not None:
        run[1] = i
    elif run is not None:
        bands2.append(run); run = None
if run:
    bands2.append(run)
print("suit x-bands (abs):", [(920 + a, 920 + b, b - a + 1) for a, b in bands2])
