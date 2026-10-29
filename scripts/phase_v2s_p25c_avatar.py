"""P25-C — 头像圆环（maroon 环）每座几何测量 → away/disc 的 avatar 锚定 ROI。"""
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


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def avatar_bbox(arr, ax, ay, tag):
    """在锚点周围 ±(260,230) 找 maroon 环（r>g+20, r>b+10, 40<r<140）。"""
    r = arr[..., 0].astype(np.int16); g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    x0, x1 = max(0, ax - 260), min(arr.shape[1], ax + 260)
    y0, y1 = max(0, ay - 230), min(arr.shape[0], ay + 220)
    rr = r[y0:y1, x0:x1]; rg = g[y0:y1, x0:x1]; rb = b[y0:y1, x0:x1]
    maroon = ((rr > rg + 20) & (rr > rb + 10) & (rr > 40) & (rr < 150))
    n = int(maroon.sum())
    if n < 200:
        print(f"{tag}: maroon n={n} (none)")
        return None
    ys, xs = np.where(maroon)
    rowd = maroon.sum(axis=1); cold = maroon.sum(axis=0)
    rows = np.where(rowd >= max(20, rowd.max() * 0.45))[0]
    cols = np.where(cold >= max(20, cold.max() * 0.45))[0]
    if not len(rows) or not len(cols):
        bx, by = x0 + int(xs.min()), y0 + int(ys.min())
        bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    else:
        bx, by = x0 + int(cols.min()), y0 + int(rows.min())
        bw = int(cols.max() - cols.min() + 1)
        bh = int(rows.max() - rows.min() + 1)
    print(f"{tag}: avatar≈({bx},{by},{bw},{bh}) n={n} "
          f"off=({bx-ax:+d},{by-ay:+d})")
    return (bx, by, bw, bh)


# 用 f03（无 overlay 遮头像）测每座
arr = load(FRAMES["f03"])
print("=== f03 每座头像圆环 ===")
for seat, (ax, ay) in ANCH.items():
    avatar_bbox(arr, ax, ay, f"f03 {seat}")
