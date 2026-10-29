"""P25-C — equity pill 精确颜色采样（f03 S2 red / f04 S1 teal / f04 S2 red）。"""
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


def sample(arr, x0, y0, w, h, tag):
    roi = arr[y0:y0 + h, x0:x0 + w]
    l = roi[..., 0] * 0.299 + roi[..., 1] * 0.587 + roi[..., 2] * 0.114
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    # 非白非暗的中间色 = pill 底色
    mid = ((l > 70) & (l < 200))
    print(f"[{tag}] mid n={int(mid.sum())} "
          f"rgb_mean=({r[mid].mean():.0f},{g[mid].mean():.0f},"
          f"{b[mid].mean():.0f})")
    # 直接看中心行
    cy = h // 2
    print(f"    row{cy}:", [
        tuple(int(v) for v in roi[cy, x][:3])
        for x in range(0, w, max(1, w // 8))])


a1 = load("POT_CHANGED_102_1787992505794_full.png")
sample(a1, 872, 184, 145, 67, "f03_S2_red_pill")
a2 = load("POT_CHANGED_13_1787992279195_full.png")
sample(a2, 195, 328, 133, 38, "f04_S1_teal_pill")
sample(a2, 871, 152, 146, 99, "f04_S2_red_pill")
