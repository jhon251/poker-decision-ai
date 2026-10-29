"""P30-2A — pot 区 + 桌中央 + 8bf8 下半（hero 需行动态）扫描。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p30_2a_crops")


def zoom(fp, x0, y0, x1, y1, name, k=1):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[y0:y1, x0:x1]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1) if k > 1 else sub
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


# f01 桌中央（pot 胶囊）
zoom("BUTTON_CHANGED_167_1787992682749_full.png", 600, 500, 1300, 800,
     "z_f01_center", 1)
# 8bf8 下半（hero 需行动？）
zoom("8bf82f134651db0add626be0304a2ff1.png", 100, 1150, 1876, 1400,
     "z_8bf8_bottom_low", 1)
# pc110 下半（hero BTN 行动态）
zoom("POT_CHANGED_110_1787992526057_full.png", 100, 1150, 1876, 1400,
     "z_pc110_bottom_low", 1)
