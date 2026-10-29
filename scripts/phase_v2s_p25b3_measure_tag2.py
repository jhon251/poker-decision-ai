"""P25-B.3 — 精确测量「盖牌」tag：暗青圆角矩形 + 白字 bbox + 相对座牌盒偏移。

从裁剪目视：tag 位于座牌盒左上角外侧偏上，f01_S5 约 x1533-1603, y890-933。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def load(frame):
    w, h, rgba = decode_png((EV / frame).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def analyze(frame, x0, y0, x1, y1, tag):
    arr = load(frame)
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    # tag 背景：暗、低饱和、偏冷（r<g 且 r<b 或近灰），luma 15-95
    dark_cool = (l > 15) & (l < 95) & (g >= r - 8) & (b >= r - 8)
    ys, xs = np.where(dark_cool)
    if len(xs) < 100:
        print(f"[{tag}] dark_cool too small: {len(xs)}")
        return
    # 取最大连通（简单：按列密度收缩）
    print(f"[{tag}] dark_cool px={len(xs)} raw bbox="
          f"({x0+xs.min()},{y0+ys.min()},{xs.max()-xs.min()+1},"
          f"{ys.max()-ys.min()+1})")
    bx0, by0, bx1, by1 = (x0+int(xs.min()), y0+int(ys.min()),
                          x0+int(xs.max()), y0+int(ys.max()))
    sub = arr[by0:by1+1, bx0:bx1+1]
    sl = sub[..., 0]*0.299 + sub[..., 1]*0.587 + sub[..., 2]*0.114
    sr, sg, sb = (sub[..., 0].astype(np.int16), sub[..., 1].astype(np.int16),
                  sub[..., 2].astype(np.int16))
    # 白字：比 tag 背景亮很多（>125）
    white = sl > 125
    wys, wxs = np.where(white)
    if len(wxs) > 0:
        print(f"    glyph bbox abs=({bx0+wxs.min()},{by0+wys.min()},"
              f"{wxs.max()-wxs.min()+1},{wys.max()-wys.min()+1}) "
              f"white px={len(wxs)}")
    # 背景色众数（在 tag 内取暗像素均值）
    bg = sub[(sl > 15) & (sl < 80)]
    if len(bg) > 50:
        print(f"    tag bg rgb_mean=({bg[:,0].mean():.0f},{bg[:,1].mean():.0f},"
              f"{bg[:,2].mean():.0f})")
    # 左右几列密度检查（圆角/边界）
    colstd = (dark_cool.sum(axis=0))
    print(f"    col density edge: first10={list(colstd[:10])} "
          f"last10={list(colstd[-10:])}")


# f01 S5 座锚 (1618,868)：tag 在其左上
analyze("BUTTON_CHANGED_167_1787992682749_full.png",
        1500, 860, 1730, 950, "f01_S5_tag")
# f01 S6 座锚 (945,1160)
analyze("BUTTON_CHANGED_167_1787992682749_full.png",
        820, 1080, 1080, 1180, "f01_S6_tag")
