"""P26-A — f05 S5 未标注负例目视（赢家动画遮挡）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p26a_crops")


def zoom(fp, x0, y0, x1, y1, name, k=4):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[y0:y1, x0:x1]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(
        encode_rgba_png(big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


zoom("POT_CHANGED_139_1787992592538_full.png", 1550, 970, 1755, 1060,
     "z_f05_S5_anim", 4)
