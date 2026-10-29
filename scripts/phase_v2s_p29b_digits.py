"""P29-B — 两卡 digit 的独立测量（行分带 + 列分带，区分卡1/卡2 digit）。"""
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
# digit 带 y 1026-1066；列投影找两个 digit 的 x 范围
roi = arr[1026:1066, 810:900]
l = roi[..., 0] * 0.299 + roi[..., 1] * 0.587 + roi[..., 2] * 0.114
ink = (l < 120).astype(np.uint8)
cold = ink.sum(axis=0)
runs, run = [], None
for i, v in enumerate(cold):
    if v >= 3 and run is None:
        run = [i, i]
    elif v >= 3 and run is not None:
        run[1] = i
    elif run is not None:
        runs.append(run); run = None
if run:
    runs.append(run)
merged = []
for rr in runs:
    if merged and rr[0] - merged[-1][1] <= 6:
        merged[-1][1] = rr[1]
    else:
        merged.append(rr)
print("digit x-bands (abs):", [(810 + a, 810 + b, b - a + 1) for a, b in merged])
# 每 band 的 y span
for a, b in merged:
    sub = ink[:, a:b + 1]
    rows = np.where(sub.sum(axis=1) > 0)[0]
    print(f"  x{810+a}-{810+b}: y {1026+rows.min()}-{1026+rows.max()}")
