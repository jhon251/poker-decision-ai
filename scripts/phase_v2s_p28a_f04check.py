"""P28-A — f04 S6 emoji pill 精确位置（为何 scan=17？）+ timer/rank 专属件。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p28a_crops")


def zoom(fp, x0, y0, x1, y1, name, k=2):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[y0:y1, x0:x1]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


zoom("POT_CHANGED_13_1787992279195_full.png", 550, 1060, 950, 1330,
     "z_f04_S6_emoji_zone", 2)
# f01 S4 rankbar 污染源（TIME 标签？）
zoom("BUTTON_CHANGED_167_1787992682749_full.png", 80, 1020, 360, 1120,
     "z_f01_S4_rankzone", 3)
