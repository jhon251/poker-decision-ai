"""P29-B — f03 c2 suit 区检查（3d 的方块是否被 poop 遮）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p29b_crops")


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


arr = load("POT_CHANGED_102_1787992505794_full.png")
sub = arr[1005:1135, 815:905]
big = np.repeat(np.repeat(sub, 5, axis=0), 5, axis=1)
(OUT / "z_f03_c2_suitzone.png").write_bytes(encode_rgba_png(
    big.shape[1], big.shape[0], big.tobytes()))
print("saved", big.shape[1], "x", big.shape[0])
