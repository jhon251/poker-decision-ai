"""HeroCards 恢复 — 第二独立帧视觉确认（corpus：POT_CHANGED_110）。
仅做裁剪 + 实际读图；身份以目视结果为准（不再用任何历史假设）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p29a_crops")


def crop(fp, x0, y0, x1, y1, name, k=3):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[y0:y1, x0:x1]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


# POT_CHANGED_110：卡区（P29-A 像素几何：白簇 813-1059, 1011-1188）
crop("POT_CHANGED_110_1787992526057_full.png", 795, 1000, 1075, 1200,
     "z_pc110_cards", 3)
