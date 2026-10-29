"""P29-B — card2 "3" digit 精确定位（窗口 855-920, 1010-1070）。"""
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
roi = arr[1010:1070, 855:925]
l = roi[..., 0] * 0.299 + roi[..., 1] * 0.587 + roi[..., 2] * 0.114
ink = (l < 120).astype(np.uint8)
cold = ink.sum(axis=0)
print("col profile x855-925 (y1010-1070):")
for i in range(0, 70, 3):
    print(f"  x{855+i}: {cold[i]}  x{855+i+1}: {cold[i+1]}")
