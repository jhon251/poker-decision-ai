"""P30-2C-A — 金额文本列段（pc114 金额行 y1294-1381 内 x1550-1800）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def segs(fid, tag, xa, xb, ya, yb):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    roi = arr[ya:yb, xa:xb]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    text = ((l > 130) & (np.abs(r - g) < 50) & (np.abs(g - b) < 50))
    cold = text.sum(axis=0)
    runs, run = [], None
    for i, v in enumerate(cold):
        if v >= 1 and run is None:
            run = [i, i]
        elif v >= 1 and run is not None:
            run[1] = i
        elif run is not None:
            runs.append(run); run = None
    if run:
        runs.append(run)
    # 合并 ≤2px
    merged = []
    for rr in runs:
        if merged and rr[0] - merged[-1][1] <= 2:
            merged[-1][1] = rr[1]
        else:
            merged.append(rr)
    print(f"[{tag}] segs: {[(xa+a, xa+b, b-a+1) for a, b in merged]}")


# pc114 金额 "124,163"（右区）
segs("POT_CHANGED_114_1787992536082_full.png", "pc114_amt", 1550, 1876, 1294, 1381)
# pc110 金额 "2,000"
segs("POT_CHANGED_110_1787992526057_full.png", "pc110_amt", 1350, 1700, 1294, 1381)
