"""P25-B.3 — 逐字符「盖」「牌」模板提取 + NCC 正负例分布。

模板来源：f01 S5 tag 内暗字（GT provenance: BUTTON_CHANGED_167, S5 folded）。
NCC：黑字二值模板 vs 每帧每座 tag 固定位置的暗二值图像；
正例=16 GT folded；负例=20 非 folded。
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


def bin_dark(arr, x0, y0, w, h):
    """tag 内暗字二值：luma < 背景均值-25（局部适应，兼容 dim 场景）。"""
    roi = arr[y0:y0 + h, x0:x0 + w]
    l = roi[..., 0] * 0.299 + roi[..., 1] * 0.587 + roi[..., 2] * 0.114
    bg = np.median(l)
    return (l < bg - 20).astype(np.float64)


# 1) 提取 f01 S5 的 盖/牌 模板（先找 tag 内暗字连通列）
arr = load(FRAMES["f01"])
bx, by, bw, bh = TAG["S5"]
binm = bin_dark(arr, bx, by, bw, bh)
cold = binm.sum(axis=0)
runs, run = [], None
for i, v in enumerate(cold):
    if v >= 2 and run is None:
        run = i
    elif v < 2 and run is not None:
        runs.append((run, i)); run = None
if run is not None:
    runs.append((run, len(cold)))
print("f01 S5 tag 内暗字列 runs:", runs)


def char_templates():
    """返回 {'gai': t, 'pai': t}——按 tag 左起第一个/第二个字形块。"""
    # 手动：两字各取稳定列范围（从 runs 数据）
    return None


def ncc(win, t):
    if t.shape[0] > win.shape[0] or t.shape[1] > win.shape[1]:
        return -1.0
    # 全窗口滑窗（小）
    th, tw = t.shape
    sub = win[:th, :tw]
    wm = sub.mean(); tm = t.mean()
    dn = np.sqrt(((sub - wm) ** 2).sum() * ((t - tm) ** 2).sum())
    if dn < 1e-9:
        return -1.0
    return float(((sub - wm) * (t - tm)).sum() / dn)


# 用整体"双字块"模板更稳：tag 左 60% 宽内容
def content_template(arr, bx, by, bw, bh, shrink=2):
    binm = bin_dark(arr, bx, by, bw, bh)
    ys, xs = np.where(binm > 0)
    if len(xs) < 20:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    # 字块 = 主字高范围；左右两字各半
    sub = binm[y0:y1 + 1, x0:x1 + 1]
    return sub[::shrink, ::shrink], (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


full_tmpl, full_box = content_template(arr, bx, by, bw, bh)
print("f01 S5 glyph block box (rel tag)=", full_box,
      "tmpl shape=", None if full_tmpl is None else full_tmpl.shape)
# 分左右两字
gai_t = full_tmpl[:, : full_tmpl.shape[1] * 2 // 3]
pai_t = full_tmpl[:, full_tmpl.shape[1] * 2 // 3:]

# 2) NCC 全矩阵
print("\nNCC (gai/pai) per frame-seat:")
print(f"{'':5s}" + "".join(f"{s:>12s}" for s in TAG))
pos_scores, neg_scores = [], []
for fr, fp in FRAMES.items():
    arr = load(fp)
    row = []
    for seat, (bx, by, bw, bh) in TAG.items():
        win = bin_dark(arr, bx, by, bw, bh)
        s = max(ncc(win, gai_t), ncc(win, pai_t))
        f = seat in GTF[fr]
        (pos_scores if f else neg_scores).append(s)
        row.append(f"{s:.2f}{'F' if f else '.'}")
    print(f"{fr:5s}" + "".join(f"{v:>12s}" for v in row))

ps = np.array(pos_scores); ns = np.array(neg_scores)
print(f"\nPOS n={len(ps)} min={ps.min():.3f} max={ps.max():.3f} "
      f"mean={ps.mean():.3f} p10={np.percentile(ps,10):.3f}")
print(f"NEG n={len(ns)} min={ns.min():.3f} max={ns.max():.3f} "
      f"mean={ns.mean():.3f} p90={np.percentile(ns,90):.3f}")
gap = ps.min() - ns.max()
print(f"gap(POS.min - NEG.max) = {gap:.3f}")
