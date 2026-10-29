"""P25-C — disc 徽章精确 bbox + 蓝色元素位置 + allin pill 精确测量。"""
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


# ── 1. disc badge（f05 S3）────────────────────────────────────────────
arr = load("POT_CHANGED_139_1787992592538_full.png")
roi = arr[300:400, 1500:1700]
r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
b = roi[..., 2].astype(np.int16)
l = r * 0.299 + g * 0.587 + b * 0.114
# 灰白字（luma 60-135, 近中性）
graytext = ((l > 60) & (l < 135) & (np.abs(r - g) < 20)
            & (np.abs(g - b) < 20))
ys, xs = np.where(graytext)
print("disc graytext n=", len(xs))
if len(xs):
    print("  bbox=", (1500 + xs.min(), 300 + ys.min(),
                      xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
# 蓝色元素 (64,90,159)
blue = ((np.abs(r - 64) < 45) & (np.abs(g - 90) < 45)
        & (np.abs(b - 159) < 55) & (b > r + 40))
ys2, xs2 = np.where(blue)
print("disc blue n=", len(xs2))
if len(xs2):
    print("  bbox=", (1500 + xs2.min(), 300 + ys2.min(),
                      xs2.max() - xs2.min() + 1, ys2.max() - ys2.min() + 1))
# 暗玻璃棒 dark glass（luma 12-45 近中性）+ 左右扩展
darkbar = (l < 45)
ys3, xs3 = np.where(darkbar)
print("disc darkbar n=", len(xs3))
if len(xs3):
    rowd = darkbar.sum(axis=1)
    rows = np.where(rowd >= max(10, rowd.max() * 0.4))[0]
    cold = darkbar.sum(axis=0)
    cols = np.where(cold >= max(10, cold.max() * 0.4))[0]
    print("  badge≈", (1500 + cols.min(), 300 + rows.min(),
                       cols.max() - cols.min() + 1,
                       rows.max() - rows.min() + 1))

# ── 2. allin pills ────────────────────────────────────────────────────
def pill_measure(fp, x0, y0, x1, y1, tag):
    a = load(fp)
    roi = a[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    red = (r > 130) & (g < 70) & (b < 70)
    teal = ((g > 100) & (g > r * 1.8) & (b > r * 1.4) & (g > b)
            & (l > 80) & (l < 190))
    blue = (b > 100) & (b > r + 40) & (b > g + 30)
    for nm, m in (("red", red), ("teal", teal), ("blue", blue)):
        n = int(m.sum())
        if n < 60:
            print(f"[{tag}] {nm}: n={n}")
            continue
        ys, xs = np.where(m)
        rowd = m.sum(axis=1)
        rows = np.where(rowd >= max(6, rowd.max() * 0.4))[0]
        cold = m.sum(axis=0)
        cols = np.where(cold >= max(6, cold.max() * 0.4))[0]
        print(f"[{tag}] {nm}: n={n} tight=({x0+xs.min()},{y0+ys.min()},"
              f"{xs.max()-xs.min()+1},{ys.max()-ys.min()+1}) "
              f"dense=({x0+cols.min()},{y0+rows.min()},"
              f"{cols.max()-cols.min()+1},{rows.max()-rows.min()+1})")
    # 白字 in pill 检查用画布


pill_measure("POT_CHANGED_102_1787992505794_full.png", 820, 150, 1060, 320,
             "f03 S2")
pill_measure("POT_CHANGED_13_1787992279195_full.png", 120, 300, 460, 470,
             "f04 S1")
pill_measure("POT_CHANGED_13_1787992279195_full.png", 820, 150, 1060, 320,
             "f04 S2")
# hero S6 f03（应无 pill：hero 亮牌无 equity）→ 负例
pill_measure("POT_CHANGED_102_1787992505794_full.png", 800, 990, 1060, 1180,
             "f03 S6 hero")
