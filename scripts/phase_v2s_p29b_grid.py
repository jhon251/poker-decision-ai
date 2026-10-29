"""P29-B — 带网格的完整卡角区放大（精确定位两卡 digit）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p29b_crops")


def grid_zoom(fp, x0, y0, x1, y1, name, k=6, step=10):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[y0:y1, x0:x1].copy()
    # 网格线（每 step px 一条黄线）
    for gx in range(0, sub.shape[1], step):
        sub[:, gx] = np.where(sub[:, gx, :3].mean(axis=1, keepdims=True)
                              > 200, sub[:, gx], [255, 220, 0, 255])
    for gy in range(0, sub.shape[0], step):
        sub[gy, :] = np.where(sub[gy, :, :3].mean(axis=1, keepdims=True)
                              > 200, sub[gy, :], [255, 220, 0, 255])
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        big.shape[1], big.shape[0], big.tobytes()))
    print(f"{name}: abs=({x0},{y0})..({x1},{y1}) grid={step}px k={k}")


grid_zoom("8bf82f134651db0add626be0304a2ff1.png", 810, 1005, 890, 1130,
          "z_8bf8_grid", 6, 10)
