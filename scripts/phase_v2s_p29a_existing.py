"""P29-A — 现有英雄卡 GT 帧视觉核查（f03 = POT_CHANGED_102, 9c 3d）
+ hero 卡区精确几何测量（双卡白面 bbox / 各卡 / 间距）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p29a_crops")
OUT.mkdir(parents=True, exist_ok=True)

FP = "POT_CHANGED_102_1787992505794_full.png"
w, h, rgba = decode_png((EV / FP).read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
print(f"frame: {w}x{h}")

# hero 卡区：白卡面（大区白像素）+ 锚点 S6 (945,1160) 上方
roi = arr[950:1200, 750:1150]
r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
b = roi[..., 2].astype(np.int16)
l = r * 0.299 + g * 0.587 + b * 0.114
white = ((l > 180) & (r > 170) & (g > 170) & (b > 170)).astype(np.uint8)
print(f"white px in (750,950)-(1150,1200): {int(white.sum())}")
# 列投影 → 双卡分离
cold = white.sum(axis=0)
runs, run = [], None
for i, v in enumerate(cold):
    if v >= 30 and run is None:
        run = [i, i]
    elif v >= 30 and run is not None:
        run[1] = i
    elif run is not None:
        runs.append(run); run = None
if run:
    runs.append(run)
print("col runs:", [(750 + a, 750 + b, b - a + 1) for a, b in runs])
# 行
rowd = white.sum(axis=1)
rows = np.where(rowd >= 30)[0]
if len(rows):
    print("row span:", 950 + rows.min(), 950 + rows.max())

# 保存放大图
sub = arr[950:1220, 750:1150]
big = np.repeat(np.repeat(sub, 2, axis=0), 2, axis=1)
(OUT / "z_f03_hero_cards.png").write_bytes(encode_rgba_png(
    big.shape[1], big.shape[0], big.tobytes()))
print("saved z_f03_hero_cards.png")
