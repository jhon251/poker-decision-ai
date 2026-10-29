"""P25-B.3 — 高倍放大「盖牌」tag 裁剪（3x nearest）供目视复核。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p25b3_crops")


def load(frame):
    w, h, rgba = decode_png((EV / frame).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def zoom(frame, x0, y0, x1, y1, name, k=4):
    arr = load(frame)
    sub = arr[y0:y1, x0:x1]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    png = encode_rgba_png(big.shape[1], big.shape[0], big.tobytes())
    (OUT / f"{name}.png").write_bytes(png)
    print(f"{name}: {big.shape[1]}x{big.shape[0]} (crop {x0},{y0},{x1},{y1})")


zoom("BUTTON_CHANGED_167_1787992682749_full.png",
     1500, 870, 1700, 990, "z_f01_S5_tag", 4)
zoom("BUTTON_CHANGED_167_1787992682749_full.png",
     820, 1080, 1020, 1200, "z_f01_S6_tag", 4)
