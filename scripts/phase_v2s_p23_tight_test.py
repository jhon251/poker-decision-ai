"""P23 — 极紧 ROI + 暗金掩码识别测试（S1/S4/S5 帧01）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import StackRecognizer  # noqa: E402
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
rec = StackRecognizer(window="1848")


def _dark_mask(self, frame):
    r = frame[..., 0].astype(np.int16)
    g = frame[..., 1].astype(np.int16)
    b = frame[..., 2].astype(np.int16)
    return ((r > 100) & (g > 70) & (r >= g) & (g >= b + 25) & (r > b + 40)
            ).astype(np.uint8)


rec._gold_mask = _dark_mask.__get__(rec, type(rec))
# 极紧 stack 行 ROI（图像测量 30px 行）
TEST = [("S1", (160, 338, 340, 372)), ("S2", (770, 282, 990, 316)),
        ("S4", (150, 1020, 350, 1054)), ("S5", (1570, 1000, 1730, 1034)),
        ("S6", (840, 1304, 1010, 1338))]
w, h, rgba = decode_png((EV /
    "BUTTON_CHANGED_167_1787992682749_full.png").read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
for seat, (x0, y0, x1, y1) in TEST:
    roi = arr[y0:y1, x0:x1]
    gold = rec._gold_mask(roi)
    res = rec.recognize_in_region(arr, (x0, y0, x1, y1))
    print(f"{seat}({x0},{y0},{x1},{y1}): gold={int(gold.sum())} "
          f"-> raw={res.get('raw_text')} value={res.get('value')} "
          f"status={res.get('status')}")
