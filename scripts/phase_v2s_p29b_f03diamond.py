"""P29-B — f03 c2 suit（红方块）精确位置：红色像素定位。"""
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
# c2 suit 区大窗 (930-1000, 1055-1130)
roi = arr[1055:1130, 930:1000]
r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
b = roi[..., 2].astype(np.int16)
red = (r > 100) & (r > g + 25) & (r > b + 25)
print(f"red px: {int(red.sum())}")
ys, xs = np.where(red)
if len(xs):
    print(f"red bbox=({930+xs.min()},{1055+ys.min()},"
          f"{xs.max()-xs.min()+1},{ys.max()-ys.min()+1})")
    # red intensity
    print(f"red rgb mean=({r[red].mean():.0f},{g[red].mean():.0f},"
          f"{b[red].mean():.0f})")
