"""P25-C — away 检测器（锚点有界搜索版）：白二值 + away 模板滑窗 NCC。
探索锚点 ± 半径。比固定偏移更鲁棒（每座头像位置已知差异）。
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
GTF = {"f01": {"S3": "away"}, "f02": {"S3": "away"}}


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def white_bin_roi(arr, x0, y0, w, h):
    x0, y0 = max(0, x0), max(0, y0)
    sub = arr[y0:y0 + h, x0:x0 + w]
    l = sub[..., 0] * 0.299 + sub[..., 1] * 0.587 + sub[..., 2] * 0.114
    r, g, b = (sub[..., 0].astype(np.int16), sub[..., 1].astype(np.int16),
               sub[..., 2].astype(np.int16))
    return ((l > 140) & (np.abs(r - g) < 30) & (np.abs(g - b) < 30)
            ).astype(np.float64)


# 模板
arr0 = load(FRAMES["f01"])
tmpl = white_bin_roi(arr0, 1585, 232, 63, 106)
ys, xs = np.where(tmpl > 0)
tmpl = tmpl[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
t0 = tmpl - tmpl.mean()
tn = float(np.sqrt((t0 * t0).sum()))


def bounded_away_ncc(arr, ax, ay, radius=170, stride=2):
    """锚点 ±radius 全窗白二值 → 模板滑窗（stride）max|NCC|。"""
    x0, y0 = ax - radius, ay - radius
    w = h = 2 * radius
    win = white_bin_roi(arr, x0, y0, w, h)
    th, tw = tmpl.shape
    if win.shape[0] < th or win.shape[1] < tw:
        return -1.0
    best = -1.0
    for dy in range(0, win.shape[0] - th + 1, stride):
        for dx in range(0, win.shape[1] - tw + 1, stride):
            sub = win[dy:dy + th, dx:dx + tw]
            s0 = sub - sub.mean()
            dn = float(np.sqrt((s0 * s0).sum()))
            if dn < 1e-9:
                continue
            v = float((s0 * t0).sum() / (dn * tn))
            best = max(best, abs(v))
    return best


print("=== away bounded-NCC matrix (radius 170) ===")
print(f"{'':5s}" + "".join(f"{s:>10s}" for s in ANCH))
pos, neg = [], []
for fr, fp in FRAMES.items():
    arr = load(fp)
    row = []
    for seat, (ax, ay) in ANCH.items():
        sc = bounded_away_ncc(arr, ax, ay)
        f = seat in GTF.get(fr, {})
        (pos if f else neg).append(sc)
        row.append(f"{sc:.2f}{'A' if f else '.'}")
    print(f"{fr:5s}" + "".join(f"{v:>10s}" for v in row))
ps, ns = np.array(pos), np.array(neg)
print(f"\nPOS n={len(ps)} min={ps.min():.3f} max={ps.max():.3f}")
print(f"NEG n={len(ns)} min={ns.min():.3f} max={ns.max():.3f}")
print(f"gap={ps.min() - ns.max():.3f}")
