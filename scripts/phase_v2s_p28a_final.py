"""P28-A — final: own-seat UI 证据总结测量（emoji pill 全帧独占 + GT 核验）。"""
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
    "f01": ("BUTTON_CHANGED_167_1787992682749_full.png", "S6"),
    "f02": ("POT_CHANGED_12_1787995290794_full.png", "S6"),
    "f03": ("POT_CHANGED_102_1787992505794_full.png", "S6"),
    "f04": ("POT_CHANGED_13_1787992279195_full.png", "S6"),
    "f05": ("POT_CHANGED_139_1787992592538_full.png", "S6"),
    "f06": ("BUTTON_CHANGED_598_1787993753587_full.png", "S6"),
    "u01": ("42b553c9468d1d1e55b9801bebad7b4f.png", "S6"),
}


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def pill_bbox(arr, ax, ay):
    """💬+😎 pill 簇（锚点 x-345..-165, y+35..+150）完整 bbox。"""
    x0, y0 = max(0, ax - 345), ay + 35
    x1, y1 = ax - 165, ay + 150
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    gold = ((r > 100) & (g > 70) & (l > 60) & (l < 210) & (r >= g))
    white = ((l > 200) & (np.abs(r - g) < 25) & (np.abs(g - b) < 30))
    m = gold | white
    ys, xs = np.where(m)
    if len(xs) < 50:
        return None, 0
    return (x0 + int(xs.min()), y0 + int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)), \
        int(m.sum())


print(f"hero=GT(6帧)S6 / 用户帧 dualang=S6")
for fr, (fp, hero_seat) in FRAMES.items():
    arr = load(fp)
    ax, ay = ANCH[hero_seat]
    bb, n = pill_bbox(arr, ax, ay)
    # 其它座检查
    other = 0
    for seat, (ox, oy) in ANCH.items():
        if seat == hero_seat:
            continue
        _, n2 = pill_bbox(arr, ox, oy)
        other += n2
    print(f"{fr:4s} hero={hero_seat} pill={'bbox='+str(bb) if bb else 'NO'} "
          f"n={n} other_seats_total={other}")
