"""P30-2C-A — 全语料下带宽度聚类 → 找到不同金额（>60px 宽带的金额值 visual 读取）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
files = sorted(EV.glob("*.png"))

groups = {}


def lower_band_profile(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    roi = arr[1150:1400, 1250:1876]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    text = ((l > 130) & (np.abs(r - g) < 50) & (np.abs(g - b) < 50))
    rowd = text.sum(axis=1)
    bs, run = [], None
    for i, v in enumerate(rowd):
        if v >= 5 and run is None:
            run = [i, i]
        elif v >= 5 and run is not None:
            run[1] = i
        elif run is not None:
            bs.append(run); run = None
    if run:
        bs.append(run)
    low = [b for b in bs if b[1] > 40]
    if not low:
        return None
    a2, b2 = low[-1]
    roi2 = roi[a2:b2 + 1]
    rg = r[a2:b2 + 1]; gg = g[a2:b2 + 1]; bb_ = b[a2:b2 + 1]
    l2 = rg * 0.299 + gg * 0.587 + bb_ * 0.114
    t2 = ((l2 > 130) & (np.abs(rg - gg) < 50) & (np.abs(gg - bb_) < 50))
    cold = t2.sum(axis=0)
    segs, run = [], None
    for i, v in enumerate(cold):
        if v >= 1 and run is None:
            run = [i, i]
        elif v >= 1 and run is not None:
            run[1] = i
        elif run is not None:
            segs.append(run); run = None
    if run:
        segs.append(run)
    widths = sorted(((b - a + 1) for a, b in segs), reverse=True)
    return widths[0] if widths else 0, int(t2.sum())


for fp in files:
    if not fp.name.endswith("_full.png"):
        continue
    try:
        w, h, rgba = decode_png(fp.read_bytes())
    except Exception:
        continue
    if (w, h) != (1876, 1400):
        continue
    got = lower_band_profile(fp.name)
    if got is None:
        continue
    maxw, ink = got
    # 粗分桶：>60 = 金额行；30-60 = 过牌/标签；<30 = X/F
    bucket = "AMOUNT" if maxw > 60 else ("CHECK" if maxw > 28 else "XF")
    groups.setdefault(bucket, []).append(fp.name)

for k, v in sorted(groups.items()):
    print(f"{k}: {len(v)} 帧")
    for s in v[:8]:
        print(f"    {s[:40]}")
