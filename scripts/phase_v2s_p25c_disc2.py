"""P25-C — disc 精确测量：玻璃棒 bbox + 断开 glyph 块结构（2字形）；
及全 36 座格在【TAG ROI 精确位置】的 glass/gray 签名。"""
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
TAG = {"S1": (138, 325, 104, 41), "S2": (816, 211, 103, 40),
       "S3": (1494, 325, 103, 41), "S4": (101, 891, 103, 40),
       "S5": (1531, 891, 103, 40), "S6": (816, 1149, 103, 40)}
GT = {"f01": {"S3": "away", "S5": "folded", "S6": "folded"},
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


# 1) disc 精确 bbox（f05 S3 大范围）
arr = load(FRAMES["f05"])
roi = arr[300:420, 1470:1700]
r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
b = roi[..., 2].astype(np.int16)
l = r * 0.299 + g * 0.587 + b * 0.114
gray = ((l >= 90) & (l <= 140) & (np.abs(r - g) < 18)
        & (np.abs(g - b) < 18))
ys, xs = np.where(gray)
print("disc gray bbox=", (1470 + xs.min(), 300 + ys.min(),
                          xs.max() - xs.min() + 1, ys.max() - ys.min() + 1),
      "n=", len(xs))
# glass bar: l<45 且 x 集中于 1494-1650
glass = l < 45
rowd = glass.sum(axis=1)
rows = np.where(rowd >= 60)[0]
print("glass dense rows=", (300 + rows.min(), 300 + rows.max()) if len(rows)
      else None)
cold = glass.sum(axis=0)
cols = np.where(cold >= 20)[0]
print("glass dense cols=", (1470 + cols.min(), 1470 + cols.max())
      if len(cols) else None)

# 2) TAG ROI 精确位置的 glass/gray 全矩阵
print("\n=== TAG ROI 内 glass/gray ===")
print(f"{'':5s} {'seat':4s} {'GT':12s} {'glass':>6s} {'gray':>5s}")
for fr, fp in FRAMES.items():
    arr = load(fp)
    for seat, (bx, by, bw, bh) in TAG.items():
        gt = GT.get(fr, {}).get(seat, "-")
        roi = arr[by:by + bh, bx:bx + bw]
        r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
        b = roi[..., 2].astype(np.int16)
        l = r * 0.299 + g * 0.587 + b * 0.114
        gl = float((l < 45).mean())
        gr = int(((l >= 90) & (l <= 140) & (np.abs(r - g) < 18)
                  & (np.abs(g - b) < 18)).sum())
        print(f"{fr:5s} {seat:4s} {gt:12s} {gl:6.2f} {gr:5d}")
