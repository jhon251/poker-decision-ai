"""P25-C — 检查 away ROI 在 f01 S1/S2 的白像素来源（误报排查）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p25c_crops")

ANCH = {"S1": (255, 310), "S2": (918, 210), "S3": (1578, 320),
        "S4": (218, 868), "S5": (1618, 868), "S6": (945, 1160)}
AWAY_OFF = (7, -88)


def zoom(fp, ax, ay, name, k=3):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    x0 = max(0, ax + AWAY_OFF[0] - 30); y0 = max(0, ay + AWAY_OFF[1] - 25)
    sub = arr[y0:y0 + 156, x0:x0 + 123]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(
        encode_rgba_png(big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0], "crop", x0, y0)


zoom("BUTTON_CHANGED_167_1787992682749_full.png", *ANCH["S1"], "z_f01_S1_awayroi")
zoom("BUTTON_CHANGED_167_1787992682749_full.png", *ANCH["S2"], "z_f01_S2_awayroi")
