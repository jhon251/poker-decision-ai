"""P30-2C-A — panel 全区网格展示（确认按钮/金额排布）——用文本段宽度可视化。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p30_2ca_crops")


def grid(fid, name):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[1150:1400, 1250:1876].copy()
    # 20px 网格线
    for gx in range(0, sub.shape[1], 20):
        sub[:, gx] = [255, 230, 60, 255]
    for gy in range(0, sub.shape[0], 20):
        sub[gy, :] = [255, 230, 60, 255]
    big = np.repeat(np.repeat(sub, 2, axis=0), 2, axis=1)
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


grid("POT_CHANGED_114_1787992536082_full.png", "z_pc114_grid")
grid("POT_CHANGED_110_1787992526057_full.png", "z_pc110_grid")
