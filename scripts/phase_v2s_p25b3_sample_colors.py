"""P25-B.3 — 精确采样：tag 背景 / tag 白字 / felt / 名字字 / 数字字 颜色。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def load(frame):
    w, h, rgba = decode_png((EV / frame).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def px(arr, x, y, tag):
    print(f"{tag}: ({x},{y}) rgb={tuple(int(v) for v in arr[y, x][:3])}")


arr = load("BUTTON_CHANGED_167_1787992682749_full.png")
# 从裁剪图估算：tag 中心 abs ≈ 1568,913；felt 左 ≈ 1448,818；
# 白字 盖牌 ≈ 1565-1596, 905-920；名字 B3nDover ≈ 1600,955；数字 3.24M ≈ 1595,1005
for x, y, t in [(1568, 913, "tag_bg?"), (1448, 818, "felt"),
                (1580, 908, "glyph_white?"), (1620, 958, "name_white?"),
                (1600, 1008, "stack_gold?"), (1450, 950, "seatbox_dark?"),
                (1700, 913, "tag_right?"), (1500, 913, "tag_left?")]:
    px(arr, x, y, t)
