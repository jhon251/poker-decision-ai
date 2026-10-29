"""HeroCards 恢复 — 8bf8 双卡精确裁剪（独立二次确认用）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

SRC = Path("data2/8bf82f134651db0add626be0304a2ff1.png")
OUT = Path("data/vision2/p29a_crops")

w, h, rgba = decode_png(SRC.read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
# 双卡 tight 区（白簇 bbox 实测）
sub = arr[1005:1195, 795:1070]
big = np.repeat(np.repeat(sub, 3, axis=0), 3, axis=1)
(OUT / "z_8bf8_cards_tight.png").write_bytes(encode_rgba_png(
    big.shape[1], big.shape[0], big.tobytes()))
print("saved", big.shape[1], "x", big.shape[0])
