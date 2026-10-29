"""P30-2C-A — 绿/炭按钮上的白字：调整掩码（亮度相对按钮底）+ 段分割。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def fin(fid, tag, xa, xb, ya, yb, gap=1):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    roi = arr[ya:yb, xa:xb]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    # 白字（绿底 g≈130-180 时白 l>200；炭底时白 l>150——统一 l>170 且低饱和）
    text = ((l > 170) & (np.abs(r - g) < 55) & (np.abs(g - b) < 55))
    rowd = text.sum(axis=1)
    rows = np.where(rowd > 0)[0]
    if len(rows) == 0:
        print(f"[{tag}] no text")
        return
    band = (rows.min(), rows.max())
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
    merged = []
    for rr in runs:
        if merged and rr[0] - merged[-1][1] <= gap:
            merged[-1][1] = rr[1]
        else:
            merged.append(rr)
    out = [(xa + a, b - a + 1, band[0], band[1] - band[0] + 1)
           for a, b in merged]
    print(f"[{tag}] segs(x,w,y,h): {out}")


fin("POT_CHANGED_114_1787992536082_full.png", "pc114 124,163",
    1400, 1600, 1330, 1400)
fin("POT_CHANGED_110_1787992526057_full.png", "pc110 2,000",
    1340, 1560, 1290, 1400)
