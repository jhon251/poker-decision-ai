"""P25-C 测量 — away/disconnected/allin 精确定位（白字/蓝/红pill/白卡面）。"""
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


def report(arr, x0, y0, x1, y1, tag):
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    white = (l > 150) & (np.abs(r - g) < 25) & (np.abs(g - b) < 25)
    blue = (b > r + 40) & (b > g + 20)
    red = (r > 140) & (g < 90) & (b < 90)
    print(f"\n== {tag} ==")
    for nm, m in (("white", white), ("blue", blue), ("red", red)):
        n = int(m.sum())
        if n < 8:
            print(f"  {nm}: {n}")
            continue
        ys, xs = np.where(m)
        print(f"  {nm}: n={n} bbox=({x0+xs.min()},{y0+ys.min()},"
              f"{xs.max()-xs.min()+1},{ys.max()-ys.min()+1})")
    return


# away f01 S3（锚 1578,320）
a = load("BUTTON_CHANGED_167_1787992682749_full.png")
report(a, 1450, 180, 1780, 430, "f01 S3 away 区")
# away f02 S3
b = load("POT_CHANGED_12_1787995290794_full.png")
report(b, 1450, 180, 1780, 430, "f02 S3 away 区")
# disc f05 S3
c = load("POT_CHANGED_139_1787992592538_full.png")
report(c, 1450, 180, 1780, 430, "f05 S3 disc 区")
# allin f03 S2（锚 918,210）
d = load("POT_CHANGED_102_1787992505794_full.png")
report(d, 780, 30, 1120, 330, "f03 S2 allin 区")
# allin f04 S1（锚 255,310）
e = load("POT_CHANGED_13_1787992279195_full.png")
report(e, 100, 150, 480, 480, "f04 S1 allin 区")
# allin f04 S2（锚 918,210）
report(e, 780, 30, 1120, 330, "f04 S2 allin 区")
