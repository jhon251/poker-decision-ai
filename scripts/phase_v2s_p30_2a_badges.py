"""P30-2A — SB/BB 徽章精确定位 + 全帧扫描（蓝 SB / 橙 BB / 底池标签）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p30_2a_crops")


def zoom(fp, x0, y0, x1, y1, name, k=4):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[y0:y1, x0:x1]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


# 8bf8 S1 座牌盒左上（SB 徽章在此附近 —— 全帧 500-600 x 550-700 区）
zoom("8bf82f134651db0add626be0304a2ff1.png", 100, 550, 500, 750,
     "z_8bf8_sbzone", 3)
# f01 S1 底池徽章（同样区）
zoom("BUTTON_CHANGED_167_1787992682749_full.png", 100, 550, 500, 750,
     "z_f01_potzone", 3)
