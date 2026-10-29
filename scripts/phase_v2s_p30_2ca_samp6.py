"""P30-2C-A — AMOUNT 组 8 帧抽样目视（找不同金额）+ XF 组 3 帧。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p30_2ca_crops")

SAMP = [
    ("POT_CHANGED_114_1787992536082_full.png", "a_pc114"),
    ("POT_CHANGED_207_1787992804365_full.png", "a_pc207"),
    ("POT_CHANGED_239_1787992880383_full.png", "a_pc239"),
    ("POT_CHANGED_322_1787993089455_full.png", "a_pc322"),
    ("POT_CHANGED_273_1787992963883_full.png", "a_pc273"),
    ("SCENE_CHANGED_144_1787992607598_full.png", "x_sc144"),
]


def zoom(fp, name):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    y1 = min(1400, h)
    sub = arr[1150:y1, 1250:1876]
    if sub.shape[0] == 0 or sub.shape[1] == 0:
        print(name, "empty")
        return
    big = np.repeat(np.repeat(sub, 2, axis=0), 2, axis=1)
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        big.shape[1], big.shape[0], big.tobytes()))
    print(name)


for fp, name in SAMP:
    zoom(fp, name)
