"""P30-2A — 盲注/ante 视觉扫描：全帧找「SB/BB/小盲/大盲/盲注」字样区域。
方法：全帧白色文本块聚类 + 中文盲注字识别（视觉）+ 候选列表。"""
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


# f01 顶部左/右边缘（盲注或桌信息常见区）
zoom("BUTTON_CHANGED_167_1787992682749_full.png", 0, 0, 500, 500,
     "z_f01_topleft", 2)
zoom("BUTTON_CHANGED_167_1787992682749_full.png", 1400, 0, 1876, 500,
     "z_f01_topright", 2)
# 8bf8 顶部左/右
zoom("8bf82f134651db0add626be0304a2ff1.png", 0, 0, 500, 500,
     "z_8bf8_topleft", 2)
zoom("8bf82f134651db0add626be0304a2ff1.png", 1400, 0, 1876, 500,
     "z_8bf8_topright", 2)
