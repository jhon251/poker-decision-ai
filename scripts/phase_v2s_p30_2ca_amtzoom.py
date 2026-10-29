"""P30-2C-A — 金额行 digit 模板提取（GT: pc114=124,163 / pc110=2,000）。

金额行位置实测：panel 坐标（减 1150,1250）：
pc114 绿钮金额行 ≈ (360-690, 150-230) → abs (1610-1940, 1300-1380)
pc110 炭钮金额行 ≈ (90-300, 190-250)   → abs (1340-1550, 1340-1400)
先定位，再 segment → 字符模板。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p30_2ca_crops")


def amount_zoom(fid, name):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[1250:1400, 1550:1876]      # 右半（金额按钮区）
    big = np.repeat(np.repeat(sub, 4, axis=0), 4, axis=1)
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


amount_zoom("POT_CHANGED_114_1787992536082_full.png", "z_pc114_amtzoom")
amount_zoom("POT_CHANGED_110_1787992526057_full.png", "z_pc110_amtzoom")
