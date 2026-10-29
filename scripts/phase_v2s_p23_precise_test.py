"""P23 — 精确 stack 区识别测试（图像测量 ROI）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import StackRecognizer  # noqa: E402
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
rec = StackRecognizer(window="1848")
TEST = [("S1", (150, 325, 360, 390)), ("S2", (760, 265, 1000, 330)),
        ("S4", (140, 1000, 360, 1070)), ("S5", (1560, 980, 1740, 1045)),
        ("S6", (830, 1290, 1020, 1350))]
w, h, rgba = decode_png((EV /
    "BUTTON_CHANGED_167_1787992682749_full.png").read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
for seat, (x0, y0, x1, y1) in TEST:
    roi = arr[y0:y1, x0:x1]
    gold = rec._gold_mask(roi)
    res = rec.recognize_in_region(arr, (x0, y0, x1, y1))
    print(f"{seat}({x0},{y0}): gold={int(gold.sum())} "
          f"-> raw={res.get('raw_text')} value={res.get('value')} "
          f"status={res.get('status')}")
