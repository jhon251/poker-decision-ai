"""P25-B.3 — debug f02 S4 + f03 暗场景：mask 命中 + 全帧亮度。"""
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


def teal_mask(arr, lo, hi):
    r = arr[..., 0].astype(np.int16); g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    return ((l > lo) & (l < hi) & (g > r + 14) & (g >= b - 4)).astype(np.uint8)


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


# 1) f02 S4
fp = "POT_CHANGED_12_1787995290794_full.png"
arr = load(fp)
m = teal_mask(arr, 30, 56)
print("f02 S4 seed(158,908) mask val =", int(m[908, 158]))
c = comp_bbox(m, (158, 908))
print("f02 S4 comp =", c)
# 不加 g>=b 约束试试
r = arr[..., 0].astype(np.int16); g = arr[..., 1].astype(np.int16)
b = arr[..., 2].astype(np.int16)
l = r * 0.299 + g * 0.587 + b * 0.114
print("f02 S4 (158,908): l=%.1f r=%d g=%d b=%d" % (l[908, 158], r[908, 158],
                                                    g[908, 158], b[908, 158]))
# 检查 (101,891)-(204,931) 区域 mask 命中率
sub = m[891:932, 101:205]
print("f02 S4 tag 区域 mask 命中率 =", float(sub.mean()))

# 2) f03 全局亮度
for fr, fp_ in (("f01", "BUTTON_CHANGED_167_1787992682749_full.png"),
                ("f02", "POT_CHANGED_12_1787995290794_full.png"),
                ("f03", "POT_CHANGED_102_1787992505794_full.png"),
                ("f04", "POT_CHANGED_13_1787992279195_full.png"),
                ("f05", "POT_CHANGED_139_1787992592538_full.png"),
                ("f06", "BUTTON_CHANGED_598_1787993753587_full.png")):
    a = load(fp_)
    ll = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    print(f"{fr} global luma mean={ll.mean():.1f} p50={np.percentile(ll,50):.1f}")

# 3) f03 S3 tag 位置像素（已知 (1494,325) 期望）
a3 = load("POT_CHANGED_102_1787992505794_full.png")
roi = a3[325:366, 1494:1597]
ll3 = roi[..., 0] * 0.299 + roi[..., 1] * 0.587 + roi[..., 2] * 0.114
print("f03 S3 tag 区域 luma: mean=%.1f p50=%.1f p90=%.1f 暗(<20)占比=%.2f" %
      (ll3.mean(), np.percentile(ll3, 50), np.percentile(ll3, 90),
       float((ll3 < 20).mean())))
