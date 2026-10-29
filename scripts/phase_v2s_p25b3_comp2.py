"""P25-B.3 — 精化 teal tag mask（luma 30-56, g>r+14, g>=b）：
全 16 折叠实例 + 12 负例 BFS 连通分量测量（尺寸/偏移/面积）。
"""
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


def teal_mask(arr):
    r = arr[..., 0].astype(np.int16); g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    return ((l > 30) & (l < 56) & (g > r + 14) & (g >= b - 4)).astype(np.uint8)


def comp_bbox(mask, seed):
    h, w = mask.shape
    if mask[seed[1], seed[0]] == 0:
        return None
    seen = np.zeros_like(mask, dtype=bool)
    stack = [seed]; seen[seed[1], seed[0]] = True
    xs, ys, n = [], [], 0
    while stack:
        x, y = stack.pop()
        n += 1; xs.append(x); ys.append(y)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (-1, -1), (1, -1), (-1, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not seen[ny, nx] \
                    and mask[ny, nx]:
                seen[ny, nx] = True; stack.append((nx, ny))
    return (min(xs), min(ys), max(xs) - min(xs) + 1,
            max(ys) - min(ys) + 1, n)


FRAMES = {
    "f01": "BUTTON_CHANGED_167_1787992682749_full.png",
    "f02": "POT_CHANGED_12_1787995290794_full.png",
    "f03": "POT_CHANGED_102_1787992505794_full.png",
    "f04": "POT_CHANGED_13_1787992279195_full.png",
    "f05": "POT_CHANGED_139_1787992592538_full.png",
    "f06": "BUTTON_CHANGED_598_1787993753587_full.png",
}
ANCH = {"S1": (255, 310), "S2": (918, 210), "S3": (1578, 320),
        "S4": (218, 868), "S5": (1618, 868), "S6": (945, 1160)}
FOLDED = [("S5", "f01"), ("S6", "f01"), ("S2", "f02"), ("S4", "f02"),
          ("S1", "f03"), ("S3", "f03"), ("S4", "f03"), ("S5", "f03"),
          ("S3", "f04"), ("S4", "f04"), ("S5", "f04"), ("S6", "f04"),
          ("S2", "f05"), ("S6", "f05"), ("S3", "f06"), ("S6", "f06")]
NEG = [("S1", "f01"), ("S3", "f01"), ("S1", "f02"), ("S5", "f02"),
       ("S2", "f03"), ("S6", "f03"), ("S1", "f04"), ("S2", "f04"),
       ("S1", "f05"), ("S4", "f05"), ("S1", "f06"), ("S2", "f06")]

# 已确认正例位置（f01/f02/f05/f06 测得）：tag 中心相对锚点
# S1:(-1447..) 其实是桌面 felt；用直接 seed 策略：锚点+(31,+5) 试探多个
SEEDS = lambda ax, ay: [(ax - 60, ay + 40), (ax - 100, ay + 20),
                        (ax - 130, ay - 10), (ax - 85, ay + 5),
                        (ax - 30, ay + 10), (ax + 30, ay + 40),
                        (ax - 120, ay + 30), (ax - 60, ay - 30),
                        (ax - 160, ay + 15), (ax + 10, ay - 20)]


def search(arr, ax, ay):
    m = teal_mask(arr)
    best = None
    for s in SEEDS(ax, ay):
        c = comp_bbox(m, s)
        if c and c[4] > 300 and (best is None or c[4] > best[4]):
            best = c
    return best


print("=== POSITIVE ===")
rows = []
for seat, fr in FOLDED:
    arr = load(FRAMES[fr]); ax, ay = ANCH[seat]
    c = search(arr, ax, ay)
    if c is None:
        print(f"{fr} {seat}: none")
        rows.append((fr, seat, None, None))
    else:
        bx, by, bw, bh, n = c
        print(f"{fr} {seat}: bbox=({bx},{by},{bw},{bh}) n={n} "
              f"off=({bx-ax:+d},{by-ay:+d})")
        rows.append((fr, seat, (bx, by, bw, bh), n))

print("=== NEGATIVE ===")
for seat, fr in NEG:
    arr = load(FRAMES[fr]); ax, ay = ANCH[seat]
    c = search(arr, ax, ay)
    if c is None:
        print(f"{fr} {seat}: clean")
    else:
        print(f"{fr} {seat}: HIT bbox={c[:4]} n={c[4]}")
