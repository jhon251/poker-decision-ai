"""P27-A — 疑似空座帧目视（S2 区 / S1 区）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p27a_crops")


def zoom(fp, x0, y0, x1, y1, name, k=2):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[y0:y1, x0:x1]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(
        encode_rgba_png(big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


# TABLE_CHANGED_166 S2 区：S2 panel ROI ≈ (845-10, 238-25, 1035+10, 385)
zoom("TABLE_CHANGED_166_1787992679506_full.png", 700, 100, 1250, 500,
     "z_tc166_S2", 2)
# POT_CHANGED_206 S1 区
zoom("POT_CHANGED_206_1787992801426_full.png", 60, 250, 500, 560,
     "z_pc206_S1", 2)
# 对照：正常帧 f01 S2
zoom("BUTTON_CHANGED_167_1787992682749_full.png", 700, 100, 1250, 500,
     "z_f01_S2_norm", 2)
