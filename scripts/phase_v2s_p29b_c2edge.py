"""P29-B — 明确分离卡与背景：用白卡面界定有效位置再测角标。
策略：窗口内仅统计"白像素周围"的墨——简化：直接找最左列墨起始。"""
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


# card2 精确左缘定位（白面起点行）
fp = "8bf82f134651db0add626be0304a2ff1.png"
arr = load(fp)
# 行 y=1050（rank 中带）扫描 x 820-900 的 luma 剖面
for y in (1030, 1050, 1090):
    row = arr[y, 820:900]
    l = row[..., 0] * 0.299 + row[..., 1] * 0.587 + row[..., 2] * 0.114
    # 找白面起点（l>150 连续）
    white_start = None
    for i, v in enumerate(l):
        if v > 150:
            white_start = 820 + i
            break
    print(f"y={y}: white_start={white_start} profile=" 
          f"{[int(v) for v in l[::5]]}")
