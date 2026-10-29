"""P29-A — 无牌 / 牌背 变体确认：看 f01（无牌暗区）与红牌背负例。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p29a_crops")


def zoom(fp, x0, y0, x1, y1, name, k=2):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[y0:y1, x0:x1]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


# f01 hero zone（无牌 → 应暗）
zoom("BUTTON_CHANGED_167_1787992682749_full.png", 700, 960, 1130, 1230,
     "z_f01_herozone_nocards", 2)
# f05 hero zone（white=478 → 什么？）
zoom("POT_CHANGED_139_1787992592538_full.png", 700, 960, 1130, 1230,
     "z_f05_herozone", 2)
