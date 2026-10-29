"""P30-2B — 精确 slot（目视校准）：stake 文本 abs x≈108-192。

1K/2K (320) 字符位置（8x zoom 推算）：
"1"≈108-114, "K"≈115-121, "/"≈122-127, "2"≈128-133, "K"≈134-140,
"("≈145-151, "3"≈152-157, "2"≈158-163, "0"≈164-169, ")"≈170-176
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/coinpoker_assets_v1876betting_header")
OUT.mkdir(parents=True, exist_ok=True)

FP = "BUTTON_CHANGED_167_1787992682749_full.png"
w, h, rgba = decode_png((EV / FP).read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
roi = arr[0:40, 0:520]
r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
b = roi[..., 2].astype(np.int16)
l = r * 0.299 + g * 0.587 + b * 0.114
mask = (l > 110) & (np.abs(r - g) < 60) & (np.abs(g - b) < 60)
# 列段（gap≤3 合并）
cold = mask.sum(axis=0)
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
    if merged and rr[0] - merged[-1][1] <= 3:
        merged[-1][1] = rr[1]
    else:
        merged.append(rr)
# stake 区段（x≥100 起，含 1K/2K (320)）
seg = [s for s in merged if s[0] >= 100]
print(f"stake 段（x≥100）: {len(seg)}")
for a, bb in seg:
    sub = mask[:, a:bb + 1]
    rows = np.where(sub.sum(axis=1) > 0)[0]
    print(f"  x{a}-{bb} w={bb-a+1} h={rows.max()-rows.min()+1}")
