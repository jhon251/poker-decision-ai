"""P29-B — f03 c2 suit（红方块）可见部分检查：红墨位置与大小。"""
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


arr = load("POT_CHANGED_102_1787992505794_full.png")
# c2 suit 区扩窗 (823-905, 1060-1130)
roi = arr[1060:1130, 823:905]
r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
b = roi[..., 2].astype(np.int16)
l = r * 0.299 + g * 0.587 + b * 0.114
red = (r > 110) & (r > g + 30) & (r > b + 30)
print(f"red px (823-905, 1060-1130): {int(red.sum())}")
ys, xs = np.where(red)
if len(xs):
    print(f"red bbox=({823+xs.min()},{1060+ys.min()},"
          f"{xs.max()-xs.min()+1},{ys.max()-ys.min()+1})")
# 查看更大窗
roi2 = arr[1040:1140, 800:910]
r2 = roi2[..., 0].astype(np.int16); g2 = roi2[..., 1].astype(np.int16)
b2 = roi2[..., 2].astype(np.int16)
red2 = (r2 > 100) & (r2 > g2 + 25) & (r2 > b2 + 25)
print(f"red px wider (800-910, 1040-1140): {int(red2.sum())}")
ys2, xs2 = np.where(red2)
if len(xs2):
    print(f"red bbox=({800+xs2.min()},{1040+ys2.min()},"
          f"{xs2.max()-xs2.min()+1},{ys2.max()-ys2.min()+1})")
