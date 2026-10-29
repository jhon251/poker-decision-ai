"""P27-B.0 — EMPTY 视觉结构精确测量（S4 用户帧）。

1) 「空的」字 bbox（窄灰字簇，与占位圈分离）
2) 幽灵圈（faint circle）几何
3) 与占用座的对照（S1 座牌盒 vs S4 空位）
4) 无牌盒确认：name/gold 全 0 的暗区 ≠ 座牌盒（dark panel 需 name+gold 佐证）
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data2/42b553c9468d1d1e55b9801bebad7b4f.png")
OUT = Path("data/vision2/p27b_crops")
w, h, rgba = decode_png(EV.read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def measure(x0, y0, x1, y1, tag):
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    gray = ((l > 90) & (l < 190) & (np.abs(r - g) < 25)
            & (np.abs(g - b) < 25))
    ys, xs = np.where(gray)
    print(f"[{tag}] gray n={len(xs)}")
    if len(xs):
        print(f"  bbox=({x0+xs.min()},{y0+ys.min()},"
              f"{xs.max()-xs.min()+1},{ys.max()-ys.min()+1})")
        # 列投影 → 字形 run
        cold = gray.sum(axis=0)
        runs, run = [], None
        for i, v in enumerate(cold):
            if v >= 2 and run is None:
                run = [i, i]
            elif v >= 2 and run is not None:
                run[1] = i
            elif run is not None:
                runs.append(run); run = None
        if run:
            runs.append(run)
        print(f"  col runs(rel): {runs}")
    return gray


# 1) 「空的」字（目视：约 (280,900)-(360,925) 帧坐标；估放大 2x crop 内 ~(310,190)-(390,240)）
# crop 是 (40,780)-(440,1050) 2x → 字在 zoom (300-390, 200-240) → frame (190-235, 880-900)
g1 = measure(150, 860, 400, 940, "empty_label(wide)")
# 更 tight：只字簇
g2 = measure(230, 870, 360, 920, "empty_label(tight)")

# 2) 幽灵圈：环形结构检查——S4 占位圈中心 ≈ 锚点 (218,868)
# 圈特征：比 felt 暗的圆环位于 felt 上方；取 (60,760)-(380,1010) 的 alt
roi = arr[760:1010, 60:380]
r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
b = roi[..., 2].astype(np.int16)
l = r * 0.299 + g * 0.587 + b * 0.114
ring = (l > 25) & (l < 48)
print(f"\n[ghost ring zone] n={int(ring.sum())} "
      f"ring density={float(ring.mean()):.3f}")
# 行向亮度剖面（圈轮廓 in 中心列 x=218-60=158）
col = l[:, 158]
print("中心列 luma 剖面(简化):", [int(v) for v in col[::20]])

# 3) 对照占用座 S5/S1 的 box 结构
measure(1600, 930, 1740, 1000, "S5_name_zone")
# 4) 确认 S4 无头像（avatar maroon circle 缺失）
sub = arr[780:960, 80:360]
r3 = sub[..., 0].astype(np.int16); g3 = sub[..., 1].astype(np.int16)
b3 = sub[..., 2].astype(np.int16)
maroon = ((r3 > g3 + 20) & (r3 > b3 + 10) & (r3 > 40) & (r3 < 150))
print(f"\n[S4 avatar zone] maroon n={int(maroon.sum())} (占用座应 >1000)")
