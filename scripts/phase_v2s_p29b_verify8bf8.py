"""HeroCards P29-B 恢复 — 为视觉确认裁剪 8bf8 帧卡区（仅几何裁剪）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p29b_crops")

# 8bf8（用户新帧）hero 卡区
fp = "8bf82f134651db0add626be0304a2ff1.png"
w, h, rgba = decode_png((EV / fp).read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
sub = arr[1000:1200, 790:1080]
big = np.repeat(np.repeat(sub, 3, axis=0), 3, axis=1)
(OUT / "z_8bf8_cards_v2.png").write_bytes(encode_rgba_png(
    big.shape[1], big.shape[0], big.tobytes()))
print("saved z_8bf8_cards_v2.png", big.shape[1], "x", big.shape[0])
