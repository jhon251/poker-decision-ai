"""P25-B.3 — 修正版 tag 签名：暗teal bg (luma 35-75, g>r+15, |g-b|<=18)
+ 内部暗字 (luma<32)。全 GT 折叠实例 vs 负例测量。"""
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
FOLDED = [("S5", "f01"), ("S6", "f01"), ("S2", "f02"), ("S4", "f02"),
          ("S1", "f03"), ("S3", "f03"), ("S4", "f03"), ("S5", "f03"),
          ("S3", "f04"), ("S4", "f04"), ("S5", "f04"), ("S6", "f04"),
          ("S2", "f05"), ("S6", "f05"), ("S3", "f06"), ("S6", "f06")]
NEG = [("S1", "f01"), ("S3", "f01"), ("S1", "f02"), ("S5", "f02"),
       ("S2", "f03"), ("S6", "f03"), ("S1", "f04"), ("S2", "f04"),
       ("S1", "f05"), ("S4", "f05"), ("S1", "f06"), ("S2", "f06")]


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def measure(arr, ax, ay):
    r = arr[..., 0].astype(np.int16); g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    x0, x1 = max(0, ax - 200), min(arr.shape[1], ax + 200)
    y0, y1 = max(0, ay - 160), min(arr.shape[0], ay + 260)
    rl = l[y0:y1, x0:x1]; rr = r[y0:y1, x0:x1]
    rg = g[y0:y1, x0:x1]; rb = b[y0:y1, x0:x1]
    bg = ((rl > 35) & (rl < 75) & (rg > rr + 15)
          & (np.abs(rg - rb) <= 18))
    if bg.sum() < 60:
        return None
    ys, xs = np.where(bg)
    rowd = bg.sum(axis=1); cold = bg.sum(axis=0)
    rows = np.where(rowd >= max(8, rowd.max() * 0.35))[0]
    cols = np.where(cold >= max(8, cold.max() * 0.35))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    bx0, by0 = x0 + int(cols.min()), y0 + int(rows.min())
    bw, bh = int(cols.max() - cols.min() + 1), int(rows.max() - rows.min() + 1)
    if bw < 30 or bh < 15 or bw > 320 or bh > 200:
        return None
    in_l = l[by0:by0 + bh, bx0:bx0 + bw]
    dark = int((in_l < 32).sum())
    return {"bbox": (bx0, by0, bw, bh), "bg_px": int(bg.sum()),
            "dark_px": dark,
            "dark_ratio": round(dark / (bw * bh), 3)}


print("=== 正例 ===")
for seat, fr in FOLDED:
    arr = load(FRAMES[fr])
    ax, ay = ANCH[seat]
    m = measure(arr, ax, ay)
    if m is None:
        print(f"{fr} {seat}: NO-mask")
    else:
        bx, by, bw, bh = m["bbox"]
        print(f"{fr} {seat}: bbox=({bx},{by},{bw},{bh}) "
              f"off=({bx-ax:+d},{by-ay:+d}) "
              f"bg={m['bg_px']} dark={m['dark_px']} "
              f"ratio={m['dark_ratio']}")

print("=== 负例 ===")
for seat, fr in NEG:
    arr = load(FRAMES[fr])
    ax, ay = ANCH[seat]
    m = measure(arr, ax, ay)
    if m is None:
        print(f"{fr} {seat}: clean")
    else:
        print(f"{fr} {seat}: HIT bg={m['bg_px']} bbox={m['bbox']} "
              f"dark={m['dark_px']} ratio={m['dark_ratio']}")
