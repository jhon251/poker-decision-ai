"""P30-2C-A — 2,000 ref 重建：pc110 金额行 y 位置确认（行剖面）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def rowprof(fid, tag, xa, xb, ya, yb):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    roi = arr[ya:yb, xa:xb]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    text = ((l > 170) & (np.abs(r - g) < 55) & (np.abs(g - b) < 55))
    rowd = text.sum(axis=1)
    print(f"[{tag}] (xa={xa}, ya={ya}):")
    for i in range(0, yb - ya, 3):
        v = int(rowd[i:i + 3].max())
        if v > 0:
            print(f"  y{ya+i}: {v}")


# pc110 金额行（跟注 2,000 炭钮：面板 x≈1474-1554 abs）
rowprof("POT_CHANGED_110_1787992526057_full.png", "pc110",
        1400, 1600, 1250, 1400)
