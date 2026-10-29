"""P30-2B — header glyph 模板提取（f01 GT 标注文本：NLH 1218466 - 1K/2K (320)）。

仅提取 stake 区（"-" 右侧）glyph：1 K / 2 K ( 3 2 0 ) —— digit {1,2,3,0} + K
+ 标点。provenance = P30-2A image-observed（V4 Flash 2026-08-31 确认文本）。
未覆盖 digit（5/9 等）/ 小数 / M 后缀 → 模板缺失 → UNKNOWN（诚实）。
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

# f01 GT 文本（P30-2A 目视）："NLH 1218466 - 1K/2K (320)"
# stake 区 = "-" 之后："1K/2K (320)"
GT_TEXT = "1K/2K (320)"
FP = "BUTTON_CHANGED_167_1787992682749_full.png"
HEADER_ROI = (0, 0, 520, 40)      # 含 hand_id 的超集（提取时右锚定）

w, h, rgba = decode_png((EV / FP).read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
x0, y0, x1, y1 = HEADER_ROI
roi = arr[y0:y1, x0:x1]
r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
b = roi[..., 2].astype(np.int16)
l = r * 0.299 + g * 0.587 + b * 0.114
white = (l > 150) & (np.abs(r - g) < 40) & (np.abs(g - b) < 40)
cold = white.sum(axis=0)
# 列 run → glyph 段（gap ≤2 并同段；数字间 gap 1-2px）
runs, run = [], None
for i, v in enumerate(cold):
    if v >= 2 and run is None:
        run = [i, i]
    elif v >= 2 and run is not None:
        run[1] = i
    elif run is not None:
        runs.append(run); run = None
if run:
    runs.append(run)
# glyph = 行带（字形连续；逗号/括号独立小段——按高度分组）
glyphs = []
for a, b in runs:
    sub = white[:, a:b+1]
    rows = np.where(sub.sum(axis=1) > 0)[0]
    if len(rows) == 0:
        continue
    gy0, gy1 = int(rows.min()), int(rows.max()) + 1
    glyphs.append((a, gy0, b - a + 1, gy1 - gy0))
print(f"glyph 段数 = {len(glyphs)}（期望 ≈ {len(GT_TEXT)} 文本字符 + NLH/手号）")
for i, (ga, gy0, gw, gh) in enumerate(glyphs):
    print(f"  glyph[{i}]: x={ga} w={gw} h={gh}")
