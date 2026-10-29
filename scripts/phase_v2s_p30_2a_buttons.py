"""P30-2A — 动作面板精确测量：
pc110（跟注 2,000）vs 8bf8（加注 4,000）—— 按钮/金额 bbox + 语义。
按钮区域右下（x 1250-1876, y 1150-1400）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p30_2a_crops")


def zoom(fp, x0, y0, x1, y1, name, k=2):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[y0:y1, x0:x1]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


zoom("POT_CHANGED_110_1787992526057_full.png", 1250, 1150, 1876, 1400,
     "z_pc110_buttons", 2)
zoom("8bf82f134651db0add626be0304a2ff1.png", 1250, 1150, 1876, 1400,
     "z_8bf8_buttons", 2)
zoom("POT_CHANGED_102_1787992505794_full.png", 1250, 1150, 1876, 1400,
     "z_f03_buttons", 2)
