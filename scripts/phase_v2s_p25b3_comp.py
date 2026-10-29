"""P25-B.3 — BFS 连通分量：从已知 tag 种子提取精确 tag 形状/bbox/offsets。"""
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
    return ((l > 30) & (l < 80) & (g > r + 14) & (g >= b - 4)).astype(np.uint8)


def component_bbox(mask, seed):
    """BFS from seed pixel; return bbox + size of that component."""
    h, w = mask.shape
    if mask[seed[1], seed[0]] == 0:
        return None
    seen = np.zeros_like(mask, dtype=bool)
    stack = [seed]
    seen[seed[1], seed[0]] = True
    xs, ys, n = [], [], 0
    while stack:
        x, y = stack.pop()
        n += 1
        xs.append(x); ys.append(y)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (-1, -1), (1, -1), (-1, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not seen[ny, nx] \
                    and mask[ny, nx]:
                seen[ny, nx] = True
                stack.append((nx, ny))
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

print("=== positive (seed = anchor + (-60,+40) 试探多个种子取大分量) ===")
for seat, fr in FOLDED:
    arr = load(FRAMES[fr])
    m = teal_mask(arr)
    ax, ay = ANCH[seat]
    best = None
    for sx, sy in ((ax - 60, ay + 40), (ax - 40, ay + 20),
                   (ax - 20, ay + 40), (ax - 80, ay + 30),
                   (ax - 60, ay + 10), (ax + 10, ay + 40)):
        c = component_bbox(m, (sx, sy))
        if c and (best is None or c[4] > best[4]):
            best = c
    if best is None:
        print(f"{fr} {seat}: no component")
        continue
    bx, by, bw, bh, n = best
    print(f"{fr} {seat}: bbox=({bx},{by},{bw},{bh}) n={n} "
          f"off=({bx-ax:+d},{by-ay:+d}) center=({bx+bw//2},{by+bh//2})")
