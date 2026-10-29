"""P25-B.3 — 点采样：tag 背景（上/下/左/右各处）、暗字、felt、座盒。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def px(arr, x, y, tag):
    v = arr[y, x][:3]
    print(f"{tag:22s} ({x:4d},{y:4d}) rgb=({v[0]:3d},{v[1]:3d},{v[2]:3d})")


arr = load("BUTTON_CHANGED_167_1787992682749_full.png")
# 由 zoom(1500,870,1700,990,4x) 推算：
# tag 左上角 ≈ zoom(125,85) → abs(1531,891)；字 盖牌 ≈ zoom(230-430,120-200)
pts = [
    (1535, 895, "tag_topleft_bg"),
    (1625, 895, "tag_topright_bg"),
    (1535, 925, "tag_botleft_bg"),
    (1600, 890, "tag_topmid_bg"),
    (1570, 912, "glyph_gai?"),
    (1605, 915, "glyph_pai?"),
    (1550, 912, "glyph_left?"),
    (1510, 890, "left_of_tag(felt)"),
    (1680, 880, "right_of_tag(felt?)"),
    (1600, 940, "below_tag(name_zone)"),
    (1600, 870, "above_tag(avatar)"),
]
for x, y, t in pts:
    px(arr, x, y, t)
