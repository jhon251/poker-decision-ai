"""P29-B — card2 左上角高倍放大（确认 rank/suit 真实位置）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p29b_crops")


def zoom(fp, x0, y0, x1, y1, name, k=6):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[y0:y1, x0:x1]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


# card2 左上 (837,1011)：放大 (835,1005)-(920,1130)
zoom("8bf82f134651db0add626be0304a2ff1.png", 835, 1005, 925, 1135,
     "z_8bf8_c2_corner_hi", 6)
# card1 左上同样
zoom("8bf82f134651db0add626be0304a2ff1.png", 810, 1005, 900, 1135,
     "z_8bf8_c1_corner_hi", 6)
