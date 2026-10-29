"""P27-B.0 — 「空的」vs「暂离」字形/位置对照测量：
两种白字文本都位于座上方——关键区分 = 座牌盒存在性（name/gold）
+「空的」字形与「暂离」字形需确认是否可像素级区分（文本模板可行性）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

FP = Path("data/vision2/coinpoker_evidence/42b553c9468d1d1e55b9801bebad7b4f.png")
OUT = Path("data/vision2/p27b_crops")
w, h, rgba = decode_png(FP.read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def white_cluster(x0, y0, x1, y1, tag):
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    white = ((l > 170) & (np.abs(r - g) < 30) & (np.abs(g - b) < 30))
    ys, xs = np.where(white)
    if len(xs) < 10:
        print(f"[{tag}] no-white n={len(xs)}")
        return None
    print(f"[{tag}] white n={len(xs)} bbox=({x0+xs.min()},{y0+ys.min()},"
          f"{xs.max()-xs.min()+1},{ys.max()-ys.min()+1}) "
          f"luma p50={np.percentile(l[white],50):.0f}")
    return (x0 + int(xs.min()), y0 + int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


# 「空的」：S4 (201,880,43,21) 目视白字（其实灰白 luma 90-190）
k = white_cluster(180, 855, 330, 925, "empty_label_white")
# 「暂离」S3：amir7777 头像左上方 ≈ (1500-1580, 790-840)?
k3 = white_cluster(1480, 770, 1660, 860, "away_S3_zhuali")
# 「暂离」S5
k5 = white_cluster(1560, 770, 1740, 880, "away_S5_zhuali")
# 咖啡杯 S3：更上
k3c = white_cluster(1520, 700, 1640, 790, "cup_S3")

# 放大两标签
for (x0, y0, x1, y1, name) in [
        (180, 850, 340, 930, "z_label_empty"),
        (1460, 740, 1680, 860, "z_label_awayS3")]:
    sub = arr[y0:y1, x0:x1]
    big = np.repeat(np.repeat(sub, 4, axis=0), 4, axis=1)
    (OUT / f"{name}.png").write_bytes(encode_rgba_png(
        big.shape[1], big.shape[0], big.tobytes()))
    print("saved", name)
