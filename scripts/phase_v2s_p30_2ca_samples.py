"""P30-2C-A — 抽样面板目视裁剪（green/red/gray/none 4 类 + 8bf8）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p30_2ca_crops")
OUT.mkdir(parents=True, exist_ok=True)


def zoom(fp, x0, y0, x1, y1, name, k=2):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[y0:y1, x0:x1]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


# green 类（过牌?）
zoom("POT_CHANGED_14_1787992281853_full.png", 1250, 1150, 1876, 1400,
     "z_green_pc14", 2)
# gray_text 类（跟注?）
zoom("POT_CHANGED_111_1787992528708_full.png", 1250, 1150, 1876, 1400,
     "z_gray_pc111", 2)
# 8bf8（red+orange+white）
zoom("8bf82f134651db0add626be0304a2ff1.png", 1250, 1150, 1876, 1400,
     "z_8bf8_buttons2", 2)
