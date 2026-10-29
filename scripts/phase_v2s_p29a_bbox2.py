"""P29-A — 卡片边界精确测量：用「暗缝」（rank/suit 区外两卡交界暗线）。
cards 直排重叠：卡1 右缘 ≈ x~925, 卡2 左缘 ~x930（目视 pc110）。
改用行带分析：白卡区域顶部两"山峰"（卡1/卡2 上缘不同 y）。
+ 测量 rank/suit 角标 bbox（供 P29-B 模板用）。"""
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


def measure(fp, tag):
    arr = load(fp)
    roi = arr[990:1195, 780:1090]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    white = ((l > 150) & (r > 140) & (g > 140) & (b > 140)).astype(np.uint8)
    # 每行白 span（顶行宽 = 两卡并排；找行 white 范围逐行）
    spans = []
    for y in range(white.shape[0]):
        xs = np.where(white[y] > 0)[0]
        if len(xs) > 30:
            spans.append((y, 780 + int(xs.min()), 780 + int(xs.max())))
    if not spans:
        print(f"[{tag}] no spans")
        return
    ytop = spans[0][0]
    # 顶沿：卡1 与 卡2 各自上缘（局部 min）
    print(f"[{tag}] first rows x-range: "
          f"y{990+ytop}: x{spans[0][1]}-{spans[0][2]}")
    # 最宽行
    widest = max(spans, key=lambda s: s[2] - s[1])
    print(f"  wide row y{990+widest[0]}: x{widest[1]}-{widest[2]} "
          f"(w={widest[2]-widest[1]+1})")
    # 角标 rank zone：x 800-860, y 1015-1060（目视 rank 左上角）
    rz = arr[1010:1075, 795:875]
    lr = rz[..., 0].astype(np.int16); gr = rz[..., 1].astype(np.int16)
    br = rz[..., 2].astype(np.int16)
    red = ((lr > 120) & (gr < 90) & (br < 90))
    dark = (lr < 80) & (gr < 80) & (br < 80)
    print(f"  rank zone red px={int(red.sum())} dark px={int(dark.sum())}")
    ys, xs = np.where(red | dark)
    if len(xs):
        print(f"  rank glyph bbox=({795+xs.min()},{1010+ys.min()},"
              f"{xs.max()-xs.min()+1},{ys.max()-ys.min()+1})")
    # suit zone：rank 下方 y 1055-1120
    sz = arr[1050:1130, 795:880]
    sr = sz[..., 0].astype(np.int16); sg = sz[..., 1].astype(np.int16)
    sb = sz[..., 2].astype(np.int16)
    s_red = ((sr > 120) & (sg < 90) & (sb < 90))
    s_dark = (sr < 80) & (sg < 80) & (sb < 80)
    ys2, xs2 = np.where(s_red | s_dark)
    if len(xs2):
        print(f"  suit glyph bbox=({795+xs2.min()},{1050+ys2.min()},"
              f"{xs2.max()-xs2.min()+1},{ys2.max()-ys2.min()+1}) "
              f"red={int(s_red.sum())} dark={int(s_dark.sum())}")


for fp, tag in [
        ("POT_CHANGED_110_1787992526057_full.png", "pc110 6c3h"),
        ("POT_CHANGED_16_1787995303501_full.png", "pc16b Kc3h")]:
    measure(fp, tag)
