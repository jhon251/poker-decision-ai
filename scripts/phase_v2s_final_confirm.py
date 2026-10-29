"""FINAL AUDIT — 说明性验证：找到的 dict-of-dicts 不一致是否为既存。
（只读：通过对比 detector 输出形状 vs compute_positions 期望形状。）"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    SeatOccupancyDetector, compute_positions)
from poker_decision_ai.vision2.pixel.realclient.seat_occupancy import (  # noqa: E402
    SeatOccupancyRecognizer1876)

EV = Path("data/vision2/coinpoker_evidence")
w, h, rgba = decode_png((EV / "BUTTON_CHANGED_167_1787992682749_full.png")
                        .read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)

# 旧 detector（1419 路径）
old = SeatOccupancyDetector(window="1848").detect(arr)
print("旧 detector 返回形状:", type(next(iter(old.values()))))
# 新 detector（1876）
new = SeatOccupancyRecognizer1876().recognize(arr)
print("新 detector 返回形状:", type(next(iter(new.values()))))

# compute_positions 期望：dict-of-strings
try:
    r = compute_positions({"S1": "OCCUPIED", "S2": "OCCUPIED",
                           "S3": "OCCUPIED", "S4": "OCCUPIED",
                           "S5": "OCCUPIED", "S6": "OCCUPIED"},
                          "S1", "S6")
    print(f"dict-of-strings → uncertainty={r['uncertainty']} "
          f"hero_position={r['hero_position']}")
except Exception as e:
    print(f"dict-of-strings error: {e}")
# dict-of-dicts 会怎样
try:
    r2 = compute_positions(old, "S1", "S6")
    print(f"dict-of-dicts → uncertainty={r2['uncertainty']} "
          f"reason={r2['reason']}")
except Exception as e:
    print(f"dict-of-dicts error: {e}")
