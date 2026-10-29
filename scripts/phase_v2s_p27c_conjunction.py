"""P27-C — EMPTY 组合判据安全性：S4-empty vs 两个 overlay 座（name/gold≈0）
vs away/disconnected。确保「name=0∧gold=0∧maroon=0∧dim字结构」不误判。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")

ANCH = {"S1": (255, 310), "S2": (918, 210), "S3": (1578, 320),
        "S4": (218, 868), "S5": (1618, 868), "S6": (945, 1160)}
STACK = {
    "S1": (165, 413, 360, 485), "S2": (845, 298, 1035, 370),
    "S3": (1520, 413, 1715, 485), "S4": (125, 978, 320, 1050),
    "S5": (1560, 978, 1745, 1050), "S6": (838, 1236, 1040, 1308),
}


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def sig(arr, seat):
    ax, ay = ANCH[seat]
    x0, y0, x1, y1 = STACK[seat]
    # name
    ln = arr[max(0, y0 - 52):y0 - 6, x0:x1]
    ln2 = ln[..., 0] * 0.299 + ln[..., 1] * 0.587 + ln[..., 2] * 0.114
    rn = ln[..., 0].astype(np.int16); gn = ln[..., 1].astype(np.int16)
    bn = ln[..., 2].astype(np.int16)
    name = int(((ln2 > 110) & (np.abs(rn - gn) < 28)
                & (np.abs(gn - bn) < 28)).sum())
    # gold
    rs = arr[y0:y1, x0:x1]
    rg = rs[..., 0].astype(np.int16); gg = rs[..., 1].astype(np.int16)
    bg2 = rs[..., 2].astype(np.int16)
    ls = rg * 0.299 + gg * 0.587 + bg2 * 0.114
    mdn = np.median(ls)
    gold = int(((rg > bg2 + 30) & (gg >= bg2 + 18) & (rg >= gg * 0.9)
                & (ls > mdn * 1.6)).sum())
    # maroon
    sub = arr[ay - 90:ay + 90, ax - 100:ax + 100]
    sr = sub[..., 0].astype(np.int16); sg = sub[..., 1].astype(np.int16)
    sb = sub[..., 2].astype(np.int16)
    maroon = int(((sr > sg + 20) & (sr > sb + 10) & (sr > 40)
                  & (sr < 150)).sum())
    # dim 字结构（锚点 ±70）：列 run ≥12px 的文本块
    lx0, ly0 = max(0, ax - 70), max(0, ay - 50)
    lx1, ly1 = min(arr.shape[1], ax + 70), min(arr.shape[0], ay + 60)
    lroi = arr[ly0:ly1, lx0:lx1]
    lr = lroi[..., 0].astype(np.int16); lg = lroi[..., 1].astype(np.int16)
    lb = lroi[..., 2].astype(np.int16)
    ll = lr * 0.299 + lg * 0.587 + lb * 0.114
    dim = ((ll >= 80) & (ll <= 190) & (np.abs(lr - lg) < 25)
           & (np.abs(lg - lb) < 25))
    dim_n = int(dim.sum())
    cold = dim.sum(axis=0)
    runs, run = [], None
    for i, v in enumerate(cold):
        if v >= 4 and run is None:
            run = [i, i]
        elif v >= 4 and run is not None:
            run[1] = i
        elif run is not None:
            runs.append(run); run = None
    if run:
        runs.append(run)
    merged = []
    for rr in runs:
        if merged and rr[0] - merged[-1][1] <= 3:
            merged[-1][1] = rr[1]
        else:
            merged.append(rr)
    max_run = max((b - a + 1 for a, b in merged), default=0)
    return name, gold, maroon, dim_n, max_run


cases = [
    ("EMPTY S4 (42b)", "42b553c9468d1d1e55b9801bebad7b4f.png", "S4"),
    ("hero overlay f03 S6", "POT_CHANGED_102_1787992505794_full.png", "S6"),
    ("anim overlay f05 S5", "POT_CHANGED_139_1787992592538_full.png", "S5"),
    ("away f01 S3", "BUTTON_CHANGED_167_1787992682749_full.png", "S3"),
    ("away f02 S3", "POT_CHANGED_12_1787995290794_full.png", "S3"),
    ("disconn f05 S3", "POT_CHANGED_139_1787992592538_full.png", "S3"),
    ("occ f01 S1", "BUTTON_CHANGED_167_1787992682749_full.png", "S1"),
    ("allin f04 S1", "POT_CHANGED_13_1787992279195_full.png", "S1"),
]
print(f"{'case':24s} {'name':>5s} {'gold':>5s} {'maroon':>6s} "
      f"{'dimN':>5s} {'run':>4s}")
for tag, fp, seat in cases:
    n, g, m, dn, mr = sig(load(fp), seat)
    print(f"{tag:24s} {n:5d} {g:5d} {m:6d} {dn:5d} {mr:4d}")
