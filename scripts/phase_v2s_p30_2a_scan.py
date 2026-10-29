"""P30-2A — 1876 帧 UI 区域扫描：hero 卡区下方/底部控制区（动作 UI 候选）
+ 桌面顶部/角落（盲注信息候选）。先裁剪全帧看点。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p30_2a_crops")


def zoom(fp, x0, y0, x1, y1, name, k=2):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[y0:y1, x0:x1]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


# f01 整帧下半（hero 动作区/底部）
zoom("BUTTON_CHANGED_167_1787992682749_full.png", 0, 1050, 1876, 1400,
     "z_f01_bottom_full", 1)
# 8bf8 下半
zoom("8bf82f134651db0add626be0304a2ff1.png", 0, 1050, 1876, 1400,
     "z_8bf8_bottom_full", 1)
# f01 顶部中央（盲注/桌信息）
zoom("BUTTON_CHANGED_167_1787992682749_full.png", 300, 0, 1600, 400,
     "z_f01_top", 1)
