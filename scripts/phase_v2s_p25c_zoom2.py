"""P25-C — f04 S1 allin 放大（对比 f03 S2：无 blue OUTS?）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p25c_crops")
fp = "POT_CHANGED_13_1787992279195_full.png"
w, h, rgba = decode_png((EV / fp).read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
sub = arr[150:480, 100:480]
big = np.repeat(np.repeat(sub, 2, axis=0), 2, axis=1)
(OUT / "z_f04_S1_allin.png").write_bytes(
    encode_rgba_png(big.shape[1], big.shape[0], big.tobytes()))
print("z_f04_S1_allin:", big.shape[1], "x", big.shape[0])
