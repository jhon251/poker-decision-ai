"""P24-B — detect_bet_capsule 内部条件调试（S5 89,600）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.coinpoker_bet import (  # noqa: E402
    gray_luma, detect_bet_capsule, CAPSULE_DARK_MAX, CAPSULE_DARK_RATIO,
    DIGIT_WHITE_MIN)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
w, h, rgba = decode_png((EV /
    "POT_CHANGED_12_1787995290794_full.png").read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
lum = gray_luma(arr)
x0, y0, ww, hh = 1350, 888, 130, 48
sub = lum[y0:y0 + hh, x0:x0 + ww]
white = (sub > DIGIT_WHITE_MIN).astype(np.uint8)
rows = white.sum(axis=1)
print(f"白像素按行: {list(rows)}")
# 暗底比（胶囊校验）
cx0, cy0 = 10, 8
cx1, cy1 = ww - 10, hh - 8
ctx = sub[cy0:cy1, cx0:cx1]
dark_ratio = float((ctx < CAPSULE_DARK_MAX).mean())
print(f"暗底比(内部 {cx0},{cy0},{cx1},{cy1}): {dark_ratio:.3f} "
      f"(需 >= {CAPSULE_DARK_RATIO})")
