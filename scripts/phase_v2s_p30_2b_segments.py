"""P30-2B — stake 区 glyph 模板提取（GT 文本 1K/2K (320)，位置标注）。

stake 区 x≈66-200, y≈4-26。文本=「1K/2K (320)」→ glyphs:
  digit-1(1K) K / digit-2(2K) K ( digit-3 digit-2 digit-0 )
模板：0,1,2,3（digits）+ K + / + ( + ) + 空格结构。
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
bg = float(np.median(l))
mask = (l > max(120.0, bg * 5)) & (np.abs(r - g) < 60) \
    & (np.abs(g - b) < 60)
# 列段（gap≤3 → 同字符）
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
print(f"bg={bg:.0f} 字符段={len(merged)}")
for a, b in merged:
    sub = mask[:, a:b+1]
    rows = np.where(sub.sum(axis=1) > 0)[0]
    if len(rows) == 0:
        continue
    print(f"  seg x{a}: w={b-a+1} h={rows.max()-rows.min()+1} "
          f"y={rows.min()}-{rows.max()} ink={int(sub.sum())}")
