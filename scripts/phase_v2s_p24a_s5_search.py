"""P24-A — S5 胶囊宽域搜索（帧04/10）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.coinpoker_bet import (  # noqa: E402
    gray_luma, detect_bet_capsule)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")

for fid in ["POT_CHANGED_12_1787995290794_full.png",
            "BUTTON_CHANGED_598_1787993753587_full.png"]:
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    lum = gray_luma(arr)
    print(f"\n=== {fid[:28]} ===")
    # S5 宽域 (1300-1876, 600-1100) 滑动 200x50
    found = []
    for yy in range(600, 1100, 15):
        for xx in range(1300, 1876 - 200, 15):
            cap = detect_bet_capsule(lum, (xx, yy, 200, 50))
            if cap is None:
                continue
            bx, by, bw, bh = cap["bbox"]
            dup = any(abs(f[0] - bx) < 50 and abs(f[1] - by) < 30 for f in found)
            if not dup:
                found.append((bx, by, bw, bh, round(cap["dark_ratio"], 2)))
    for f in found[:8]:
        print(f"  胶囊 ({f[0]},{f[1]},{f[2]},{f[3]}) dark={f[4]}")
