"""P26-A — 核查两处"gold 命中"：
1) f01 S1 (138,325,103,41)：是否橙色底池 label？
2) f01 S5 (1531,891,104,40)：teal tag 为何过 gold 掩码（g>=b+25？）
   输出：gold 掩码叠加图 + 像素分布。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p26a_crops")


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def gold_mask(roi):
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    return ((r > 100) & (g > 70) & (r >= g)
            & (g >= b + 25) & (r > b + 40))


def report(arr, bx, by, bw, bh, name):
    roi = arr[by:by + bh, bx:bx + bw]
    m = gold_mask(roi)
    r = roi[..., 0]; g = roi[..., 1]; b = roi[..., 2]
    hit = roi[np.where(m)]
    print(f"[{name}] gold_n={int(m.sum())}/{bw*bh}")
    if len(hit):
        print(f"    hit rgb mean=({hit[:,0].mean():.0f},{hit[:,1].mean():.0f},"
              f"{hit[:,2].mean():.0f})")
        # 哪些行命中
        rows = np.where(m.sum(axis=1) > 0)[0]
        print(f"    hit rows: {by+rows.min()}-{by+rows.max()}")
    # 保存叠加图
    ov = roi.copy()
    ov[m] = (255, 0, 255, 255)
    big = np.repeat(np.repeat(ov, 4, axis=0), 4, axis=1)
    (OUT / f"gold_ov_{name}.png").write_bytes(
        encode_rgba_png(big.shape[1], big.shape[0], big.tobytes()))
    print(f"    saved gold_ov_{name}.png")


a1 = load("BUTTON_CHANGED_167_1787992682749_full.png")
report(a1, 138, 325, 103, 41, "f01_S1_roi")
report(a1, 1531, 891, 104, 40, "f01_S5_roi")
