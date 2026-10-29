"""P25-B.3 — 全矩阵：6帧×6座 在每座固定 tag 位置的「teal tag」签名测量。

固定 tag bbox（已从多帧验证，每座常量）：
S1 (138,325,104,41) S2 (816,211,103,40) S3 (1494,325,103,41)
S4 (101,891,103,40) S5 (1531,891,103,40) S6 (816,1149,103,40)
签名 = teal mask (g>1.5r, |g-b|<=14, 6<l<110) 覆盖率 + 内部暗字占比。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")

FRAMES = {
    "f01": "BUTTON_CHANGED_167_1787992682749_full.png",
    "f02": "POT_CHANGED_12_1787995290794_full.png",
    "f03": "POT_CHANGED_102_1787992505794_full.png",
    "f04": "POT_CHANGED_13_1787992279195_full.png",
    "f05": "POT_CHANGED_139_1787992592538_full.png",
    "f06": "BUTTON_CHANGED_598_1787993753587_full.png",
}
TAG = {"S1": (138, 325, 104, 41), "S2": (816, 211, 103, 40),
       "S3": (1494, 325, 103, 41), "S4": (101, 891, 103, 40),
       "S5": (1531, 891, 103, 40), "S6": (816, 1149, 103, 40)}
# GT folded per frame
GTF = {"f01": {"S5", "S6"}, "f02": {"S2", "S4"},
       "f03": {"S1", "S3", "S4", "S5"}, "f04": {"S3", "S4", "S5", "S6"},
       "f05": {"S2", "S6"}, "f06": {"S3", "S6"}}


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def score(arr, bx, by, bw, bh):
    roi = arr[by:by + bh, bx:bx + bw]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    teal = ((g > r * 1.5) & (np.abs(g - b) <= 14) & (l > 6) & (l < 110))
    dark = l < 32
    return float(teal.mean()), float(dark.mean())


print(f"{'':5s}" + "".join(f"{s:>12s}" for s in TAG))
for fr, fp in FRAMES.items():
    arr = load(fp)
    row = []
    for seat, (bx, by, bw, bh) in TAG.items():
        t, d = score(arr, bx, by, bw, bh)
        f = seat in GTF[fr]
        tagv = "F" if f else "."
        row.append(f"({t:.2f}/{d:.2f}){tagv}")
    print(f"{fr:5s}" + "".join(f"{v:>12s}" for v in row))
