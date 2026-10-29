"""P30-2D-A — 全帧候选金额 UI 盘点：除 pot/座椅 capsule/面板按钮外，
是否还有任何独立"当前下注额"显示。颜色/文本块聚类（排除已知对象）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p30_2da_crops")


def zoom(fp, x0, y0, x1, y1, name, k=3):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    y1 = min(y1, h)
    sub = arr[y0:y1, x0:x1]
    if sub.shape[0] == 0 or sub.shape[1] == 0:
        print(name, "empty")
        return
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


# pc110 座间 bet capsule（S1 1,000 / S2 2,000）——检查其位置与样式
zoom("POT_CHANGED_110_1787992526057_full.png", 440, 430, 800, 560,
     "z_pc110_bets1", 3)
# pc110 各座 capsule 全貌（左上区）
zoom("POT_CHANGED_110_1787992526057_full.png", 380, 380, 1100, 700,
     "z_pc110_bets_wide", 2)
