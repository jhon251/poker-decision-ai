"""P28-A — emoji pill 精确定位（S6 hero 左侧）+ 全帧扫描。

从 z_42b_S6_ownui（crop 600,1050,1200,1400 @2x）推算：
💬 在 zoom(30-190, 330-500) → frame(615-695, 1215-1300)
😎 在 zoom(200-300, 340-440) → frame(700-750, 1220-1270)
→ hero emoji zone = (600, 1195, 780, 1310)，相对 S6 锚 (945,1160)：
   x -345..-165, y +35..+150。
对每座用同一相对偏移量测（校验独占性）。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")

ANCH = {"S1": (255, 310), "S2": (918, 210), "S3": (1578, 320),
        "S4": (218, 868), "S5": (1618, 868), "S6": (945, 1160)}
FRAMES = {
    "f01": "BUTTON_CHANGED_167_1787992682749_full.png",
    "f02": "POT_CHANGED_12_1787995290794_full.png",
    "f03": "POT_CHANGED_102_1787992505794_full.png",
    "f04": "POT_CHANGED_13_1787992279195_full.png",
    "f05": "POT_CHANGED_139_1787992592538_full.png",
    "f06": "BUTTON_CHANGED_598_1787993753587_full.png",
    "u01": "42b553c9468d1d1e55b9801bebad7b4f.png",
}


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def emoji_zone_sig(arr, ax, ay):
    """相对偏移 zone：x ax-345..-165, y ay+35..+150（S6 实测推导）。"""
    x0, y0 = max(0, ax - 345), min(arr.shape[0] - 1, ay + 35)
    x1, y1 = ax - 165, min(arr.shape[0], ay + 150)
    roi = arr[y0:y1, x0:x1]
    if roi.size == 0:
        return 0, 0
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    yellow = ((r > 150) & (g > 100) & (b < 120) & (r >= g)).sum()
    l = r * 0.299 + g * 0.587 + b * 0.114
    white = ((l > 200) & (np.abs(r - g) < 25) & (np.abs(g - b) < 30)).sum()
    return int(yellow), int(white)


print(f"{'':4s}" + "".join(f"{s:>10s}" for s in ANCH))
for fr, fp in FRAMES.items():
    arr = load(fp)
    row = []
    for seat, (ax, ay) in ANCH.items():
        y, w = emoji_zone_sig(arr, ax, ay)
        row.append(f"{y}/{w}")
    print(f"{fr:4s}" + "".join(f"{v:>10s}" for v in row))
