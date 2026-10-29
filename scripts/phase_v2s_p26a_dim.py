"""P26-A — DIM 场景 gold 文本实测：颜色 / 相对判据 / ROI 内是否仍可分割。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")

STACK_ROWS_1876 = {
    "S1": (165, 410, 375, 485),
    "S2": (845, 295, 1035, 375),
    "S3": (1560, 410, 1680, 485),
    "S4": (108, 975, 345, 1055),
    "S5": (1555, 975, 1725, 1050),
    "S6": (835, 1225, 1055, 1315),
}


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def probe(fp, seat, value, tag):
    arr = load(fp)
    x0, y0, x1, y1 = STACK_ROWS_1876[seat]
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    # 与 felt/背景的差异：找"亮于背景"的像素
    bg = np.median(l)
    bright = l > bg * 1.5
    print(f"[{tag} {seat} {value}] bg_luma={bg:.1f} "
          f"bright_n={int(bright.sum())}")
    if bright.sum():
        ys, xs = np.where(bright)
        hit = roi[np.where(bright)]
        print(f"    bright bbox=({x0+xs.min()},{y0+ys.min()},"
              f"{xs.max()-xs.min()+1},{ys.max()-ys.min()+1}) "
              f"rgb=({hit[:,0].mean():.0f},{hit[:,1].mean():.0f},"
              f"{hit[:,2].mean():.0f})")
    # gold-relative: r > b+40 & g >= b+25（绝对阈值去掉）
    rel = ((r > b + 40) & (g >= b + 25) & (r >= g) & (l > bg * 1.3))
    print(f"    rel_gold_n={int(rel.sum())}")
    if rel.sum():
        ys, xs = np.where(rel)
        print(f"    rel bbox=({x0+xs.min()},{y0+ys.min()},"
              f"{xs.max()-xs.min()+1},{ys.max()-ys.min()+1})")


# DIM 正例们（f03=f02? 实际帧：f03=POT_CHANGED_102, f04=POT_CHANGED_13）
probe("POT_CHANGED_102_1787992505794_full.png", "S1", 721840, "f03")
probe("POT_CHANGED_102_1787992505794_full.png", "S5", 2230000, "f03")
probe("POT_CHANGED_102_1787992505794_full.png", "S3", 423772, "f03")
probe("POT_CHANGED_13_1787992279195_full.png", "S4", 379760, "f04")
probe("POT_CHANGED_13_1787992279195_full.png", "S6", 184440, "f04")
probe("POT_CHANGED_13_1787992279195_full.png", "S3", 148440, "f04")
# NORMAL 对照
probe("BUTTON_CHANGED_167_1787992682749_full.png", "S1", 101600, "f01")
