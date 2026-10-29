"""P25-C — disc 文本结构判据：gray 像素列 run（≥12 宽, ≥2 个, 间隙≥3）。
对比 f05 S3（正例）vs f06 S1 wolf / f02 S6 / f01 S4 / f01 S3 (away)。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")

TAG = {"S1": (138, 325, 104, 41), "S2": (816, 211, 103, 40),
       "S3": (1494, 325, 103, 41), "S4": (101, 891, 103, 40),
       "S5": (1531, 891, 103, 40), "S6": (816, 1149, 103, 40)}


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def text_struct(arr, bx, by, bw, bh, tag):
    roi = arr[by:by + bh, bx:bx + bw]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    gray = ((l >= 90) & (l <= 140) & (np.abs(r - g) < 18)
            & (np.abs(g - b) < 18))
    glass = (l < 45).mean()
    cold = gray.sum(axis=0)
    runs, run = [], None
    for i, v in enumerate(cold):
        if v >= 6 and run is None:
            run = [i, i]
        elif v >= 6 and run is not None:
            run[1] = i
        else:
            if run is not None:
                runs.append(run); run = None
    if run:
        runs.append(run)
    # 合并 ≤3px 间隙
    merged = []
    for rr in runs:
        if merged and rr[0] - merged[-1][1] <= 3:
            merged[-1][1] = rr[1]
        else:
            merged.append(rr)
    wide = [(a, b) for (a, b) in merged if b - a + 1 >= 12]
    print(f"[{tag}] glass={glass:.2f} gray_n={int(gray.sum())} "
          f"runs={[(a, b) for (a, b) in merged]} wide>=12: {len(wide)}")


arr = load("POT_CHANGED_139_1787992592538_full.png")
text_struct(arr, *TAG["S3"], "f05 S3 DISC")
arr = load("BUTTON_CHANGED_598_1787993753587_full.png")
text_struct(arr, *TAG["S1"], "f06 S1 wolf")
arr = load("POT_CHANGED_12_1787995290794_full.png")
text_struct(arr, *TAG["S6"], "f02 S6")
arr = load("BUTTON_CHANGED_167_1787992682749_full.png")
text_struct(arr, *TAG["S4"], "f01 S4")
text_struct(arr, *TAG["S3"], "f01 S3 AWAY")
text_struct(arr, *TAG["S5"], "f01 S5 folded")
arr = load("POT_CHANGED_102_1787992505794_full.png")
text_struct(arr, *TAG["S6"], "f03 S6")
arr = load("POT_CHANGED_139_1787992592538_full.png")
text_struct(arr, *TAG["S5"], "f05 S5")
