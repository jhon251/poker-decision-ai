"""P30-2B — header 阈值优化：luma>100 (bg*6) 抗噪分割 + token 几何。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def seg(fp, tag):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    roi = arr[0:40, 0:520]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    bg = float(np.median(l))
    mask = (l > max(100.0, bg * 6)) & (np.abs(r - g) < 60) \
        & (np.abs(g - b) < 60)
    cold = mask.sum(axis=0)
    runs, run = [], None
    for i, v in enumerate(cold):
        if v > 1 and run is None:
            run = [i, i]
        elif v > 1 and run is not None:
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
    print(f"[{tag}] bg={bg:.0f} mask_px={int(mask.sum())} 字符段={len(merged)}")
    for a, b in merged:
        sub = mask[:, a:b+1]
        rows = np.where(sub.sum(axis=1) > 0)[0]
        if len(rows) == 0:
            continue
        print(f"    x{a}: w={b-a+1} h={rows.max()-rows.min()+1} "
              f"ink={int(sub.sum())}")


for fp in ["BUTTON_CHANGED_167_1787992682749_full.png",
           "8bf82f134651db0add626be0304a2ff1.png",
           "POT_CHANGED_110_1787992526057_full.png"]:
    seg(fp, fp[:6])
