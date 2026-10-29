"""P29-B — 完整 digit 带测量（y 1005-1070, x 810-905）逐行投影找两个 digit。"""
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
roi = arr[1005:1070, 810:905]
l = roi[..., 0] * 0.299 + roi[..., 1] * 0.587 + roi[..., 2] * 0.114
ink = (l < 120).astype(np.uint8)
# 列投影（排除卡缘线 y<1014 行）
ink_c = ink[10:, :]   # y 1015+
cold = ink_c.sum(axis=0)
print("col profile (y1015-1070):")
for i in range(0, 95, 5):
    print(f"  x{810+i}: {cold[i]}  x{810+i+2}: {cold[i+2]}")
