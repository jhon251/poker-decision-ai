"""P30-2B — header 字符分组提取（gap≤3 合并 → 14px 字符）+ 模板保存。"""
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

FP = "BUTTON_CHANGED_167_1787992682749_full.png"   # GT: NLH 1218466 - 1K/2K (320)
w, h, rgba = decode_png((EV / FP).read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
roi = arr[0:40, 0:520]
r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
b = roi[..., 2].astype(np.int16)
l = r * 0.299 + g * 0.587 + b * 0.114
white = (l > 150) & (np.abs(r - g) < 40) & (np.abs(g - b) < 40)
cold = white.sum(axis=0)
# 字符列带：黑列（无墨）为分隔——gap ≥2 空白列分割
runs, run = [], None
for i, v in enumerate(cold):
    if v > 0 and run is None:
        run = [i, i]
    elif v > 0 and run is not None:
        run[1] = i
    elif run is not None:
        runs.append(run); run = None
if run:
    runs.append(run)
# 合并 ≤3px 空隙（字符内部笔划间 gap）
merged = []
for rr in runs:
    if merged and rr[0] - merged[-1][1] <= 3:
        merged[-1][1] = rr[1]
    else:
        merged.append(rr)
print(f"字符段 = {len(merged)}")
labels = "NLH 1218466 - 1K/2K (320)"   # 11 可见字符 + 空格
chars = []
for a, b in merged:
    sub = white[:, a:b+1]
    rows = np.where(sub.sum(axis=1) > 0)[0]
    if len(rows) == 0:
        continue
    gy0, gy1 = int(rows.min()), int(rows.max()) + 1
    chars.append((a, gy0, b - a + 1, gy1 - gy0, int(white[gy0:gy1, a:b+1].sum())))
for i, c in enumerate(chars):
    print(f"  char[{i}]: x={c[0]} w={c[2]} h={c[3]} ink={c[4]}")
print(f"\nGT 字符序: {[ch for ch in labels if ch != ' ']} ({sum(1 for ch in labels if ch != ' ')} 字符)")
