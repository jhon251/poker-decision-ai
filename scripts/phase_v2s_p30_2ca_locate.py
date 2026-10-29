"""P30-2C-A — 金额行精确定位（panel 相对坐标 → abs）：
pc114: 在 z_pc114_amount (crop 1250,1240,1600,1400 @3x) 中金额行
zoom(530-900, 310-420) → abs(1250+530/3≈1427-1550, 1240+103-1380)。
重新实测：定位文字 bbox。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def locate_amount(fid, tag):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    # panel abs ROI (1250,1150)-(1876,1400)
    panel = arr[1150:1400, 1250:1876]
    r = panel[..., 0].astype(np.int16); g = panel[..., 1].astype(np.int16)
    b = panel[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    text = ((l > 130) & (np.abs(r - g) < 50) & (np.abs(g - b) < 50))
    rowd = text.sum(axis=1)
    bands, run = [], None
    for i, v in enumerate(rowd):
        if v >= 5 and run is None:
            run = [i, i]
        elif v >= 5 and run is not None:
            run[1] = i
        elif run is not None:
            bands.append(run); run = None
    if run:
        bands.append(run)
    print(f"[{tag}] 行带: {[(a, b, b-a+1) for a, b in bands]}")
    low = [b for b in bands if b[1] > 40]
    if low:
        a2, b2 = low[-1]
        roi = panel[a2:b2 + 1]
        l2 = l[a2:b2 + 1]
        t2 = ((l2 > 130) & (np.abs(r[a2:b2+1] - g[a2:b2+1]) < 50)
              & (np.abs(g[a2:b2+1] - b[a2:b2+1]) < 50))
        cold = t2.sum(axis=0)
        xs = np.where(cold > 0)[0]
        if len(xs):
            print(f"  lower band abs bbox=({1250+xs.min()},{1150+a2},"
                  f"{xs.max()-xs.min()+1},{b2-a2+1})")


locate_amount("POT_CHANGED_114_1787992536082_full.png", "pc114 124,163")
locate_amount("POT_CHANGED_110_1787992526057_full.png", "pc110 2,000")
