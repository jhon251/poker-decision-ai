"""P25-B.3 — folded「盖牌」逐字符定位 + 正负例 NCC 分布（最小）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _chars(frame, x0, y0, x1, y1):
    """区域内中灰文字字符段（luma 60-170 亮簇）。"""
    w, h, rgba = decode_png((EV / frame).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    m = (l > 60) & (l < 180)
    cols = m.sum(axis=0)
    runs, run = [], None
    for i, v in enumerate(cols):
        if v >= 2 and run is None:
            run = i
        elif v < 2 and run is not None:
            runs.append((run, i))
            run = None
    if run is not None:
        runs.append((run, len(cols)))
    out = []
    for (cs, ce) in runs:
        seg = m[:, cs:ce]
        rys = np.where(seg.sum(axis=1) > 0)[0]
        if len(rys) < 8:
            continue
        out.append((x0 + cs, y0 + rys.min(), ce - cs,
                    rys.max() - rys.min() + 1))
    return out


# 帧01 S5「盖牌」区（B3nDover 座牌盒上缘，P25-B.2 发现 y950-968）
print("帧01 S5 盖牌字符段:")
for c in _chars("BUTTON_CHANGED_167_1787992682749_full.png",
                1530, 940, 1700, 975):
    print(f"  char bbox=({c[0]},{c[1]},{c[2]},{c[3]})")
# 帧01 S5 玩家名（负例近邻）
print("帧01 S5 玩家名区（负例）:")
for c in _chars("BUTTON_CHANGED_167_1787992682749_full.png",
                1560, 975, 1780, 1010):
    print(f"  char bbox=({c[0]},{c[1]},{c[2]},{c[3]})")
