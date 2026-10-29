"""P25-C — 最终判据测量：
A) disc: TAG ROI 内灰白字 (luma 90-140 中性) 计数
B) allin: "白字被彩色包夹" 行数（flanked-white ≥8, ≥20 连续行）
C) 汇总：18 正例签名 vs 负例
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


def disc_gray(arr, bx, by, bw, bh):
    roi = arr[by:by + bh, bx:bx + bw]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    return int(((l >= 90) & (l <= 140) & (np.abs(r - g) < 18)
                & (np.abs(g - b) < 18)).sum())


def pill_flanked(arr, ax, ay):
    x0, y0 = max(0, ax - 150), max(0, ay - 140)
    x1, y1 = min(arr.shape[1], ax + 150), min(arr.shape[0], ay + 140)
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    red = ((r > 150) & (g < 95) & (b < 95)).astype(np.int16)
    teal = ((g > 120) & (g > r + 90) & (g > b + 20)).astype(np.int16)
    blue = ((b > 100) & (b > r + 40) & (b > g + 30)).astype(np.int16)
    white = ((l > 190) & (r > 180) & (g > 180) & (b > 180)).astype(np.int16)
    # allow white>=200 (dim 场景近似)
    col = red | teal | blue
    W = 25
    height = col.shape[0]
    flank_rows = 0
    best_run = 0
    run = 0
    for y in range(col.shape[1] // 1 if False else height):
        row_c = col[y]
        row_w = white[y]
        # 白像素左右 25px 内都有彩
        cnt = 0
        idxw = np.flatnonzero(row_w)
        for i in idxw:
            a = max(0, i - W); bb = min(row_c.shape[0], i + W + 1)
            if row_c[a:i].sum() >= 3 and row_c[i + 1:bb].sum() >= 3:
                cnt += 1
        if cnt >= 8 and int(row_c.sum()) >= 60:
            run += 1
            best_run = max(best_run, run)
        else:
            run = 0
    return best_run, int(col.sum())


print(f"{'':5s} {'seat':4s} {'GT':12s} {'discGray':>8s} {'pillRun':>7s} "
      f"{'colN':>6s}")
for fr, fp in FRAMES.items():
    arr = load(fp)
    for seat, (ax, ay) in ANCH.items():
        gt = GT.get(fr, {}).get(seat, "-")
        dg = disc_gray(arr, *TAG[seat])
        pr, cn = pill_flanked(arr, ax, ay)
        print(f"{fr:5s} {seat:4s} {gt:12s} {dg:8d} {pr:7d} {cn:6d}")
