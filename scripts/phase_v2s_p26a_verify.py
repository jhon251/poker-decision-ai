"""P26-A — 目视核查：
1) f01 S1 (138,325,103,41) —— gold 掩码命中 3610px 的是什么东西？
2) f01 S5 (1531,891,103,40) —— 同一位置（S5 folded tag）为何也命中？
3) 每座真实 stack 行（gold 数字行）放大图（取一帧）
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p26a_crops")
OUT.mkdir(parents=True, exist_ok=True)


def zoom(fp, x0, y0, x1, y1, name, k=4):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[y0:y1, x0:x1]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(
        encode_rgba_png(big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


# 1) f01 S1 的 (138,325,103,41) 及其周边（含座牌盒上缘）
zoom("BUTTON_CHANGED_167_1787992682749_full.png", 60, 280, 420, 490,
     "z_f01_S1_goldsrc", 3)
# 2) f01 S5 的 (1531,891,103,40)（folded tag 位置）
zoom("BUTTON_CHANGED_167_1787992682749_full.png", 1450, 840, 1780, 1060,
     "z_f01_S5_goldsrc", 3)
# 3) 真实 stack 行：S1 (130-360, 400-470)、S6 (860-1040, 1220-1300)
zoom("BUTTON_CHANGED_167_1787992682749_full.png", 130, 395, 380, 480,
     "z_f01_S1_stackrow", 4)
zoom("BUTTON_CHANGED_167_1787992682749_full.png", 850, 1215, 1060, 1300,
     "z_f01_S6_stackrow", 4)
