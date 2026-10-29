"""P25-C 测量 — 目视确认 disc/away 区域放大图。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p25c_crops")
OUT.mkdir(parents=True, exist_ok=True)


def zoom(fp, x0, y0, x1, y1, name, k=3):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[y0:y1, x0:x1]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(
        encode_rgba_png(big.shape[1], big.shape[0], big.tobytes()))
    print(f"{name}: {big.shape[1]}x{big.shape[0]} crop=({x0},{y0},{x1},{y1})")


zoom("POT_CHANGED_139_1787992592538_full.png",
     1500, 200, 1720, 380, "z_f05_S3_disc", 4)
zoom("BUTTON_CHANGED_167_1787992682749_full.png",
     1540, 200, 1700, 360, "z_f01_S3_away", 4)
zoom("POT_CHANGED_102_1787992505794_full.png",
     780, 30, 1120, 330, "z_f03_S2_allin", 2)
