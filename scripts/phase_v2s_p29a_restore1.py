"""HeroCards 证据恢复 — 从 8bf8 新帧裁剪 hero 卡区（仅几何裁剪，不作身份判断）。
卡区参考 = P29-A 合法保留的像素几何（真实测量）：S6 卡区 ≈ (780,990)-(1090,1190)。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

SRC = Path("data2/8bf82f134651db0add626be0304a2ff1.png")
OUT = Path("data/vision2/p29a_crops")

w, h, rgba = decode_png(SRC.read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
print(f"frame: {w}x{h}")

# 1) hero 卡区裁剪（完整）
sub = arr[950:1250, 700:1150]
big = np.repeat(np.repeat(sub, 2, axis=0), 2, axis=1)
(OUT / "z_8bf8_herozone.png").write_bytes(encode_rgba_png(
    big.shape[1], big.shape[0], big.tobytes()))
print("saved z_8bf8_herozone.png", big.shape[1], "x", big.shape[0])

# 2) 双卡白簇测量（真实像素量，不代表身份）
roi = arr[990:1195, 780:1090]
r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
b = roi[..., 2].astype(np.int16)
l = r * 0.299 + g * 0.587 + b * 0.114
white = ((l > 150) & (r > 140) & (g > 140) & (b > 140))
print(f"hero zone white px: {int(white.sum())}")
