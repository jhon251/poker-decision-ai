"""P29-B — 独立测量 card1 与 card2 的角标 offset（三帧一致）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")

# card 名义 bbox
C1 = (813, 1011, 121, 178)
C2 = (837, 1011, 122, 178)


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def corner_bands(arr, card, tag):
    cx, cy = card[0], card[1]
    # 卡左上 130x130（覆盖角标区）
    roi = arr[cy - 5:cy + 130, cx - 8:cx + 100]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    ink = l < 120
    rowd = ink.sum(axis=1)
    runs, run = [], None
    for i, v in enumerate(rowd):
        if v >= 3 and run is None:
            run = [i, i]
        elif v >= 3 and run is not None:
            run[1] = i
        elif run is not None:
            runs.append(run); run = None
    if run:
        runs.append(run)
    merged = []
    for rr in runs:
        if merged and rr[0] - merged[-1][1] <= 4:
            merged[-1][1] = rr[1]
        else:
            merged.append(rr)
    bands = [(a - 5, b - 5, b - a + 1) for a, b in merged]  # rel card 左上
    print(f"[{tag}] bands rel=({cx},{cy}): {bands}")
    # 每个 band 的 x 范围
    for a, b, _ in bands[:3]:
        sub = ink[a + 5:b + 6]
        xs = np.where(sub.sum(axis=0) > 0)[0]
        if len(xs):
            print(f"    band y{a}..{b}: x rel {xs.min()-8}..{xs.max()-8}")


for fp, tag in [
        ("8bf82f134651db0add626be0304a2ff1.png", "8bf8"),
        ("POT_CHANGED_110_1787992526057_full.png", "pc110")]:
    arr = load(fp)
    print(f"=== {tag} ===")
    corner_bands(arr, C1, f"{tag} c1")
    corner_bands(arr, C2, f"{tag} c2")
