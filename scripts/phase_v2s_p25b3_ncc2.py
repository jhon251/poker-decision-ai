"""P25-B.3 — 修正版 NCC：滑窗式 + dim 场景自适应二值化。
验证「盖牌」字形在 16 正例 tag 位置高匹配、20 负例不匹配。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")

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
GTF = {"f01": {"S5", "S6"}, "f02": {"S2", "S4"},
       "f03": {"S1", "S3", "S4", "S5"}, "f04": {"S3", "S4", "S5", "S6"},
       "f05": {"S2", "S6"}, "f06": {"S3", "S6"}}


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def bin_dark_arr(arr, bx, by, bw, bh):
    roi = arr[by:by + bh, bx:bx + bw]
    l = roi[..., 0] * 0.299 + roi[..., 1] * 0.587 + roi[..., 2] * 0.114
    bg = np.median(l)
    return (l < bg - 20).astype(np.float64)


def slide_ncc(win, t):
    """win >= t：滑窗取 max |NCC|（处理字形偏移）。"""
    th, tw = t.shape
    if win.shape[0] < th or win.shape[1] < tw:
        return -1.0
    t0 = t - t.mean()
    tn = float(np.sqrt((t0 * t0).sum()))
    if tn < 1e-9:
        return -1.0
    best = -1.0
    for dy in range(win.shape[0] - th + 1):
        for dx in range(win.shape[1] - tw + 1):
            sub = win[dy:dy + th, dx:dx + tw]
            s0 = sub - sub.mean()
            dn = float(np.sqrt((s0 * s0).sum()))
            if dn < 1e-9:
                continue
            v = float((s0 * t0).sum() / (dn * tn))
            best = max(best, abs(v))
    return best


# 模板：f01 S5 tag 内暗字左右两块（列 runs (28,50),(53,77)）
arr = load(FRAMES["f01"])
bx, by, bw, bh = TAG["S5"]
binm = bin_dark_arr(arr, bx, by, bw, bh)
ys, xs = np.where(binm > 0)
y0, y1 = ys.min(), ys.max()
# 左字 27..50, 右字 53..77（runs 扩展1px）
gai_t = binm[y0:y1 + 1, 26:51]
pai_t = binm[y0:y1 + 1, 52:78]
print(f"gai_t={gai_t.shape} pai_t={pai_t.shape} "
      f"(y {y0}-{y1}, 高={y1-y0+1})")

print("\nNCC(滑窗 max|·|) matrix:")
print(f"{'':5s}" + "".join(f"{s:>12s}" for s in TAG))
pos, neg = [], []
for fr, fp in FRAMES.items():
    arr = load(fp)
    row = []
    for seat, (bx, by, bw, bh) in TAG.items():
        win = bin_dark_arr(arr, bx, by, bw, bh)
        s = max(slide_ncc(win, gai_t), slide_ncc(win, pai_t))
        f = seat in GTF[fr]
        (pos if f else neg).append(s)
        row.append(f"{s:.2f}{'F' if f else '.'}")
    print(f"{fr:5s}" + "".join(f"{v:>12s}" for v in row))

ps, ns = np.array(pos), np.array(neg)
print(f"\nPOS n={len(ps)} min={ps.min():.3f} max={ps.max():.3f} "
      f"mean={ps.mean():.3f} p10={np.percentile(ps, 10):.3f}")
print(f"NEG n={len(ns)} min={ns.min():.3f} max={ns.max():.3f} "
      f"mean={ns.mean():.3f} p90={np.percentile(ns, 90):.3f}")
print(f"gap(POS.min - NEG.max) = {ps.min() - ns.max():.3f}")
