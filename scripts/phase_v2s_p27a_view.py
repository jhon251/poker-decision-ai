"""P27-A — 目视核查 empty-like 帧：BUTTON_CHANGED_168 / POT_CHANGED_109 /
POT_CHANGED_10（S1=0 或 S3-S6=0）。全座截图。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p27a_crops")
OUT.mkdir(parents=True, exist_ok=True)


def zoom(fp, x0, y0, x1, y1, name, k=2):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[y0:y1, x0:x1]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(
        encode_rgba_png(big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


# 左半区（S1/S4 + S2 顶部）
zoom("BUTTON_CHANGED_168_1787992686302_full.png", 0, 0, 940, 1400,
     "z_f168_left", 1)
# 右半区
zoom("BUTTON_CHANGED_168_1787992686302_full.png", 940, 0, 1876, 1400,
     "z_f168_right", 1)
# POT_CHANGED_10 全桌
zoom("POT_CHANGED_10_1787992271691_full.png", 0, 0, 1876, 1400,
     "z_fpc10_full", 1)
