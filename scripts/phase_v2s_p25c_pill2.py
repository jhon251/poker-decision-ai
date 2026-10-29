"""P25-C — equity pill 确定判据：行内 color-run 宽度（>=80）且带白字。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")

ANCH = {"S1": (255, 310), "S2": (918, 210), "S3": (1578, 320),
        "S4": (218, 868), "S5": (1618, 868), "S6": (945, 1160)}
FRAMES = {
    "f01": "BUTTON_CHANGED_167_1787992682749_full.png",
    "f02": "POT_CHANGED_12_1787995290794_full.png",
    "f03": "POT_CHANGED_102_1787992505794_full.png",
    "f04": "POT_CHANGED_13_1787992279195_full.png",
    "f05": "POT_CHANGED_139_1787992592538_full.png",
    "f06": "BUTTON_CHANGED_598_1787993753587_full.png",
}
GT = {"f01": {"S3": "away", "S5": "folded", "S6": "folded"},
      "f02": {"S2": "folded", "S3": "away", "S4": "folded"},
      "f03": {"S1": "folded", "S2": "allin", "S3": "folded", "S4": "folded",
              "S5": "folded"},
      "f04": {"S1": "allin", "S2": "allin", "S3": "folded", "S4": "folded",
              "S5": "folded", "S6": "folded"},
      "f05": {"S2": "folded", "S3": "disconnected", "S6": "folded"},
      "f06": {"S3": "folded", "S6": "folded"}}


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def pill_rows(arr, ax, ay):
    x0, y0 = max(0, ax - 150), max(0, ay - 140)
    x1, y1 = min(arr.shape[1], ax + 150), min(arr.shape[0], ay + 140)
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    # 饱和红 OR 中饱和绿(teal pill 44,186,149)
    red = (r > 150) & (g < 95) & (b < 95)
    teal = ((g > 120) & (g > r + 90) & (g > b + 20))
    white = ((l > 200) & (r > 190) & (g > 190) & (b > 180))
    best = None
    for nm, m in (("red", red), ("teal", teal)):
        # 每行最大连续 run 宽
        rowsw = []
        for row in m:
            # fast: cumsum
            cs = np.concatenate(([0], np.cumsum(row.astype(np.int32))))
            width = 0
            # 最大 run：position of ones
            idx = np.flatnonzero(row)
            if len(idx) >= 60:
                # run length via diff
                gaps = np.diff(idx)
                # split
                breaks = np.flatnonzero(gaps > 2)
                starts = np.concatenate(([0], breaks + 1))
                ends = np.concatenate((breaks, [len(idx) - 1]))
                width = int(max(ends - starts + 1))
            rowsw.append(width)
        rowsw = np.array(rowsw)
        wide = np.flatnonzero(rowsw >= 80)
        if not len(wide):
            continue
        # 连续区间
        runs, run = [], None
        prev = None
        for rw in wide:
            if run is None:
                run = [rw, rw]
            elif rw - run[1] <= 2:
                run[1] = rw
            else:
                runs.append(run); run = [rw, rw]
        if run:
            runs.append(run)
        for (a, bb) in runs:
            bh = bb - a + 1
            if not (18 <= bh <= 100):
                continue
            wtxt = int(white[a:bb + 1].sum())
            if wtxt < 40:
                continue
            if best is None or bh > best[1]:
                best = (nm, bh, wtxt)
    return best


print(f"{'':5s} {'seat':4s} {'GT':8s} pill")
for fr, fp in FRAMES.items():
    arr = load(fp)
    for seat, (ax, ay) in ANCH.items():
        gt = GT.get(fr, {}).get(seat, "-")
        p = pill_rows(arr, ax, ay)
        mark = " ALLIN" if gt == "allin" else ""
        print(f"{fr:5s} {seat:4s} {gt:8s} {p}{mark}")
