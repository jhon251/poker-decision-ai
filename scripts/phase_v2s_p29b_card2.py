"""P29-B — card2 角标实际位置测量（相对 card2 左上 (837,1011)）。
扫描 card2 左上区 (837..900, 1011..1090) 的暗/红墨 tight bbox，
与 RANK_OFF/SUIT_OFF 对比。"""
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


def scan_card2(fp, tag):
    arr = load(fp)
    # card2 左上 90x130 区
    roi = arr[1008:1140, 830:930]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    ink = l < 120
    # 列投影→ rank/suit 分带（行）
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
    # 合并≤4px 间隙
    merged = []
    for rr in runs:
        if merged and rr[0] - merged[-1][1] <= 4:
            merged[-1][1] = rr[1]
        else:
            merged.append(rr)
    print(f"[{tag}] ink row bands (rel 830,1008): "
          f"{[(a, b, b-a+1) for a, b in merged]}")
    # 每带 x bbox
    for a, b in merged[:2]:
        sub = ink[a:b+1]
        xs = np.where(sub.sum(axis=0) > 0)[0]
        if len(xs):
            print(f"   band y{1008+a}-{1008+b}: x{830+xs.min()}-{830+xs.max()} "
                  f"(rel card2: y{a-3}, x{xs.min()-7})")


for fp, tag in [
        ("8bf82f134651db0add626be0304a2ff1.png", "8bf8(c2=3c)"),
        ("POT_CHANGED_110_1787992526057_full.png", "pc110(c2=3h)"),
        ("POT_CHANGED_102_1787992505794_full.png", "f03(c2=3d)")]:
    scan_card2(fp, tag)
