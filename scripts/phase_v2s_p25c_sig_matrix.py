"""P25-C — 全 36 座格签名测量：disc灰字 / pill / teal / away。
为生产检测器定阈值（只从测量取值，不调低至伪造）。"""
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


def graytext_in(arr, x0, y0, w, h):
    roi = arr[y0:y0 + h, x0:x0 + w]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    return int(((l > 60) & (l < 135) & (np.abs(r - g) < 20)
                & (np.abs(g - b) < 20)).sum())


# disc ROI：tag ROI 加宽（玻璃棒更宽）+ 上扩
DISC_ROI = {"S1": (138, 315, 135, 60), "S2": (816, 201, 135, 60),
            "S3": (1494, 315, 135, 60), "S4": (101, 881, 135, 60),
            "S5": (1531, 881, 135, 60), "S6": (816, 1139, 135, 60)}


def teal_cov(arr, bx, by, bw, bh):
    roi = arr[by:by + bh, bx:bx + bw]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    return float(((g > r * 1.5) & (np.abs(g - b) <= 14)
                  & (l > 6) & (l < 110)).mean())


def pill_sig(arr, ax, ay):
    """锚点 ±130 窗口找 equity pill（宽饱和红/teal 圆角矩形带白字）。"""
    x0, y0 = max(0, ax - 130), max(0, ay - 130)
    x1, y1 = min(arr.shape[1], ax + 130), min(arr.shape[0], ay + 130)
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    red = (r > 130) & (g < 70) & (b < 70)
    teal = ((g > 100) & (g > r * 1.8) & (b > r * 1.4) & (g > b)
            & (l > 80) & (l < 190))
    blue = (b > 100) & (b > r + 40) & (b > g + 30)
    res = {}
    for nm, m in (("red", red), ("teal", teal), ("blue", blue)):
        n = int(m.sum())
        if n < 150:
            res[nm] = (0, 0)
            continue
        rowd = m.sum(axis=1); cold = m.sum(axis=0)
        rows = np.where(rowd >= max(6, rowd.max() * 0.4))[0]
        cols = np.where(cold >= max(6, cold.max() * 0.4))[0]
        if not len(rows) or not len(cols):
            res[nm] = (0, 0)
            continue
        bw2 = int(cols.max() - cols.min() + 1)
        bh2 = int(rows.max() - rows.min() + 1)
        res[nm] = (n, bw2 if bw2 >= bh2 else bh2)
    return res


print(f"{'':5s} {'seat':4s} {'GT':11s} {'teal':>5s} {'gray':>5s} "
      f"{'red':>6s} {'tealP':>6s} {'blue':>6s}")
for fr, fp in FRAMES.items():
    arr = load(fp)
    for seat, (ax, ay) in ANCH.items():
        gt = GT.get(fr, {}).get(seat, "-")
        t = teal_cov(arr, *TAG[seat])
        g = graytext_in(arr, *DISC_ROI[seat])
        p = pill_sig(arr, ax, ay)
        print(f"{fr:5s} {seat:4s} {gt:11s} {t:5.2f} {g:5d} "
              f"{p['red'][0]:6d} {p['teal'][0]:6d} {p['blue'][0]:6d}")
