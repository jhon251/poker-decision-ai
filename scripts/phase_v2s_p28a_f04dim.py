"""P28-A — f04 DIM 变体：relaxed mask 验证 emoji pill 存在性（目视已见）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


arr = load("POT_CHANGED_13_1787992279195_full.png")
# f04 S6 pill 区（zoom 推算 abs 550-800, 1210-1310）
roi = arr[1210:1310, 550:800]
r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
b = roi[..., 2].astype(np.int16)
l = r * 0.299 + g * 0.587 + b * 0.114
bg = np.median(l)
print(f"f04 pill 区 bg_luma={bg:.1f} max={l.max():.1f}")
# 相对判据：亮度 > 背景×1.3（DIM 鲁棒）+ 色差
rel = ((l > bg * 1.3) & (np.abs(r - g) < 45)).astype(np.uint8)
print(f"rel 像素: n={int(rel.sum())}")
ys, xs = np.where(rel > 0)
if len(xs):
    print(f"  rel bbox=({550+xs.min()},{1210+ys.min()},"
          f"{xs.max()-xs.min()+1},{ys.max()-ys.min()+1})")
    hit = roi[ys, xs]
    print(f"  rgb mean=({hit[:,0].mean():.0f},{hit[:,1].mean():.0f},"
          f"{hit[:,2].mean():.0f})")
# 直接打印 zoom 目视坐标的像素：😎 (~700-775, 1240-1300)
for (x, y) in [(700, 1250), (720, 1260), (750, 1260), (660, 1250)]:
    print(f"  ({x},{y}) rgb={tuple(int(v) for v in arr[y, x][:3])} "
          f"luma={l[y-1210, x-550]:.0f}")
