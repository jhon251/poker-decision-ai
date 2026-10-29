"""P28-A — 修正 emoji 掩码（gold 脸 r>100,g>70, 60<l<210, r>=g）+ 全矩阵。
Emoji pill zone = 座锚 x-345..-165, y+35..+150（已目视校准：S6 💬(620-700)
+ 😎(706-775) 在锚 (945,1160) 左下）。
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


def pill_sig(arr, ax, ay):
    x0, y0 = max(0, ax - 345), ay + 35
    x1, y1 = ax - 165, ay + 150
    roi = arr[y0:y1, x0:x1]
    if roi.size == 0:
        return 0, 0, 0
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    gold = ((r > 100) & (g > 70) & (l > 60) & (l < 210) & (r >= g))
    white = ((l > 200) & (np.abs(r - g) < 25) & (np.abs(g - b) < 30))
    red = ((r > 140) & (g < 70) & (b < 70))
    return int(gold.sum()), int(white.sum()), int(red.sum())


print(f"{'':4s}" + "".join(f"{s:>12s}" for s in ANCH))
for fr, fp in FRAMES.items():
    arr = load(fp)
    row = []
    for seat, (ax, ay) in ANCH.items():
        g_, w_, r_ = pill_sig(arr, ax, ay)
        row.append(f"{g_}/{w_}/{r_}")
    print(f"{fr:4s}" + "".join(f"{v:>12s}" for v in row))
