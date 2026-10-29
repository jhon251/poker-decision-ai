"""P28-A — 😎 emoji 精确颜色采样（f01 S6 / f03 S6）。"""
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


def sample(fp, x0, y0, x1, y1, tag):
    arr = load(fp)
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    # 金色 face：非白非黑
    l = r * 0.299 + g * 0.587 + b * 0.114
    gold = ((r > 100) & (g > 70) & (l > 60) & (l < 210))
    ys, xs = np.where(gold)
    print(f"[{tag}] gold-ish n={len(xs)}")
    if len(xs):
        print(f"    bbox=({x0+xs.min()},{y0+ys.min()},"
              f"{xs.max()-xs.min()+1},{ys.max()-ys.min()+1}) "
              f"rgb=({r[gold].mean():.0f},{g[gold].mean():.0f},"
              f"{b[gold].mean():.0f})")
        # 中心采样
        cx, cy = int(xs.mean()), int(ys.mean())
        print(f"    center({x0+cx},{y0+cy}) rgb="
              f"{tuple(int(v) for v in roi[cy, cx][:3])}")


# 😎 f01 S6: abs ≈ (706-773, 1220-1300)
sample("BUTTON_CHANGED_167_1787992682749_full.png", 700, 1215, 780, 1305,
       "f01_S6_emoji")
# 😎 f03 S6
sample("POT_CHANGED_102_1787992505794_full.png", 700, 1215, 780, 1305,
       "f03_S6_emoji")
# 💬 f01 S6
sample("BUTTON_CHANGED_167_1787992682749_full.png", 620, 1215, 700, 1305,
       "f01_S6_chat")
