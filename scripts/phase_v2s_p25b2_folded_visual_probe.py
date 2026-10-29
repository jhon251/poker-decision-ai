"""P25-B.2 — folded 徽章 ASCII 颜色图（帧01 S5 B3nDover 座牌盒区）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
w, h, rgba = decode_png((EV /
    "BUTTON_CHANGED_167_1787992682749_full.png").read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
# B3nDover 座牌盒宽域（P25-A 目视 盖牌 在盒左上）
x0, y0, x1, y1 = 1520, 890, 1790, 970
roi = arr[y0:y1, x0:x1]
r = roi[..., 0].astype(np.int16)
g = roi[..., 1].astype(np.int16)
b = roi[..., 2].astype(np.int16)
l = r * 0.299 + g * 0.587 + b * 0.114
print(f"区 ({x0},{y0},{x1},{y1}) 颜色图: W=白>170  G=绿(g>r+15)  C=青(bl/g高)  "
      "D=暗  .=其它")
for yy in range(0, y1 - y0, 6):
    row = ""
    for xx in range(0, x1 - x0, 6):
        rr2 = r[yy, xx]; gg2 = g[yy, xx]; bb2 = b[yy, xx]; ll2 = l[yy, xx]
        if ll2 > 170:
            row += "W"
        elif gg2 > rr2 + 15 and gg2 > 60:
            row += "G"
        elif bb2 > rr2 + 20 and gg2 > 60:
            row += "C"
        elif ll2 < 60:
            row += "D"
        else:
            row += "."
    print(f"  y{y0+yy}: {row}")
