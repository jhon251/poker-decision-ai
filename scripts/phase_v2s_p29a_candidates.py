"""P29-A — 候选帧目视核查（不同 cluster / variant）：
A) (813-1059) 全卡显示：POT_CHANGED_110 / POT_CHANGED_16_5303501 / POT_CHANGED_143
B) (939-1009) 窄带：POT_CHANGED_109 / POT_CHANGED_15
C) (835-1077) 变异：SCENE_CHANGED_74
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p29a_crops")


def zoom(fp, x0, y0, x1, y1, name, k=2):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[y0:y1, x0:x1]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


# 全部裁剪 hero 卡区 (700,960)-(1130,1230)
for fp, tag in [
        ("POT_CHANGED_110_1787992526057_full.png", "z_pc110"),
        ("POT_CHANGED_16_1787995303501_full.png", "z_pc16b"),
        ("POT_CHANGED_143_1787992603846_full.png", "z_pc143"),
        ("POT_CHANGED_109_1787992523050_full.png", "z_pc109"),
        ("POT_CHANGED_15_1787995299856_full.png", "z_pc15"),
        ("SCENE_CHANGED_74_1787992438991_full.png", "z_sc74")]:
    zoom(fp, 700, 960, 1130, 1230, tag, 2)
