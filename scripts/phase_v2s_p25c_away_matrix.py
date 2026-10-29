"""P25-C — away/disconnected/allin 精确 ROI 测量（每座恒定布局常量）。"""
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
}
GTF = {"f01": {"S3": "away", "S5": "folded", "S6": "folded"},
       "f02": {"S2": "folded", "S3": "away", "S4": "folded"},
       "f03": {"S1": "folded", "S2": "allin", "S3": "folded", "S4": "folded",
               "S5": "folded"},
       "f04": {"S1": "allin", "S2": "allin", "S3": "folded", "S4": "folded",
               "S5": "folded", "S6": "folded"},
       "f05": {"S2": "folded", "S3": "disconnected", "S6": "folded"},
       "f06": {"S3": "folded", "S6": "folded"}}


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


# away ROI：头像圆上（白字+图标）。f01 S3 实测 white bbox=(1585,232,63,106)
# 相对锚 (1578,320)：off=(+7,-88)。取每座锚点 + off ± 余量 (110,130,110,140)
AWAY_OFF = (7, -88)
def away_roi(ax, ay):
    return (ax + AWAY_OFF[0] - 30, ay + AWAY_OFF[1] - 25,
            63 + 60, 106 + 50)


def measure(arr, ax, ay, tag):
    x0, y0, w, h = away_roi(ax, ay)
    x0, y0 = max(0, x0), max(0, y0)
    roi = arr[y0:y0 + h, x0:x0 + w]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    white = (l > 150) & (np.abs(r - g) < 25) & (np.abs(g - b) < 25)
    blue = (b > r + 40) & (b > g + 30)
    nw, nb = int(white.sum()), int(blue.sum())
    gy = ("away" if (nw > 150 and nb < 20) else
          ("disc" if nw > 150 and nb >= 20 else "-"))
    print(f"{tag:14s} white={nw:5d} blue={nb:5d} -> {gy}")


print("=== away 区（白/蓝）全矩阵 ===")
for fr, fp in FRAMES.items():
    arr = load(fp)
    for seat, (ax, ay) in ANCH.items():
        measure(arr, ax, ay, f"{fr} {seat}")
