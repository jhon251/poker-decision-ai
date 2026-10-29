"""P29-B — 全卡区网格放大（815-1090, 1000-1200）定位两卡所有 glyph。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p29b_crops")


def grid_zoom(fp, x0, y0, x1, y1, name, k=4, step=20):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[y0:y1, x0:x1].copy()
    for gx in range(0, sub.shape[1], step):
        sub[:, gx] = [255, 230, 60, 255]
    for gy in range(0, sub.shape[0], step):
        sub[gy, :] = [255, 230, 60, 255]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        big.shape[1], big.shape[0], big.tobytes()))
    print(f"{name}: abs=({x0},{y0}) grid={step}px")


grid_zoom("8bf82f134651db0add626be0304a2ff1.png", 810, 1000, 1090, 1200,
          "z_8bf8_fullgrid", 4, 20)
