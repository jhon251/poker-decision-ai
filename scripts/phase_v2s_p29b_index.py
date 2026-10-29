"""P29-B — 单卡左上角 index（rank + 小 suit）高倍放大目视。"""
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


# card1 左上角 (813,1011) 区：x 813-870, y 1011-1090
zoom("8bf82f134651db0add626be0304a2ff1.png", 813, 1008, 880, 1090,
     "z_8bf8_c1_index", 6)
zoom("POT_CHANGED_110_1787992526057_full.png", 837, 1008, 905, 1090,
     "z_pc110_c2_index", 6)
