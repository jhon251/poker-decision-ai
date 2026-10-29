"""P28-A — S5 zone 白色污染核查 + hero 专属件（表情 pill/手牌力条）精确 bbox。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p28a_crops")


def zoom(fp, x0, y0, x1, y1, name, k=3):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[y0:y1, x0:x1]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


# f02 S5 白色来源（bet 胶囊？）
zoom("POT_CHANGED_12_1787995290794_full.png", 1260, 880, 1480, 1030,
     "z_f02_S5_zone", 3)
# S6 emoji pill 精确 bbox (f01)
zoom("BUTTON_CHANGED_167_1787992682749_full.png", 580, 1180, 800, 1330,
     "z_f01_S6_emoji", 3)
# 手牌力条+rank (f04)
zoom("POT_CHANGED_13_1787992279195_full.png", 700, 1150, 1000, 1300,
     "z_f04_S6_rankbar", 3)
