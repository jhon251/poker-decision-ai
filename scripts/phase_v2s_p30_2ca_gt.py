"""P30-2C-A — 金额行 GT 记录（pc114=124,163 / pc110=2,000 / pc177=check）+ 数量扫描。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p30_2ca_crops")
GT = Path("data/vision2/coinpoker_1876_gt_to_call.json")


def zoom(fid, x0, y0, x1, y1, name, k=3):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[y0:y1, x0:x1]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


# 金额行定位（pc114 跟注 124,163：按钮在 (0-450, 1240-1400) 区）
zoom("POT_CHANGED_114_1787992536082_full.png", 1250, 1240, 1600, 1400,
     "z_pc114_amount", 3)
zoom("POT_CHANGED_110_1787992526057_full.png", 1350, 1150, 1650, 1400,
     "z_pc110_amount", 3)
