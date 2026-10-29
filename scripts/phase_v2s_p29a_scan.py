"""P29-A — 全 1876 语料扫描：hero 卡区出现两枚白卡面的帧（候选英雄牌样本）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")

# hero 卡区（P15 实测 810-1050 × 1005-1165；放宽到 780-1090 × 990-1180）
X0, Y0, X1, Y1 = 780, 990, 1090, 1180


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


files = sorted(EV.glob("*.png"))
cands = []
scanned = 0
for fp in files:
    if not fp.name.endswith("_full.png"):
        continue
    try:
        w, h, rgba = decode_png(fp.read_bytes())
    except Exception:
        continue
    if (w, h) != (1876, 1400):
        continue
    scanned += 1
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    roi = arr[Y0:Y1, X0:X1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    white = ((l > 150) & (r > 140) & (g > 140) & (b > 140))
    # 候选：白卡面成对（列投影两簇各 ≥40px 宽）
    cold = white.sum(axis=0)
    runs, run = [], None
    for i, v in enumerate(cold):
        if v >= 25 and run is None:
            run = [i, i]
        elif v >= 25 and run is not None:
            run[1] = i
        elif run is not None:
            runs.append(run); run = None
    if run:
        runs.append(run)
    # 合并 ≤8px 间隙（双卡紧邻）
    merged = []
    for rr in runs:
        if merged and rr[0] - merged[-1][1] <= 8:
            merged[-1][1] = rr[1]
        else:
            merged.append(rr)
    wide = [m for m in merged if m[1] - m[0] + 1 >= 45]
    rowd = white.sum(axis=1)
    tall = int((rowd >= 30).sum())
    if len(wide) >= 1 and tall >= 60:
        cands.append((fp.name, [(X0 + a, X0 + b) for a, b in wide], tall,
                      int(white.sum())))
print(f"scanned={scanned} candidates={len(cands)}")
for name, spans, tall, wn in cands:
    print(f"  {name}: spans={spans} tall={tall} white={wn}")
