"""P28-A — hero rank 元件（进度条 rail + 一对/两对 文本）全矩阵 + 等待盲注按钮。"""
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
    "u01": "42b553c9468d1d1e55b9801bebad7b4f.png",
}


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def hero_rank_sig(arr, ax, ay):
    """锚点正下方 y+180..+260 的橙黄 rank rail（f03 目视：一对/两对+bar）。"""
    x0, y0 = max(0, ax - 150), ay + 175
    x1, y1 = ax + 150, ay + 265
    roi = arr[y0:y1, x0:x1]
    if roi.size == 0:
        return 0, 0
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    orange = ((r > 130) & (g > 70) & (b < 110) & (r >= g))
    gray_text = ((l > 60) & (l < 190) & (np.abs(r - g) < 30)
                 & (np.abs(g - b) < 30))
    return int(orange.sum()), int(gray_text.sum())


# 等待盲注按钮（右下，仅用户帧 hero 可见；绿色圆角按钮）
def wait_button(arr):
    x0, y0 = 1550, 1180
    x1, y1 = 1876, 1330
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    green = ((g > 110) & (g > r + 30) & (g > b + 20)).sum()
    return int(green)


print(f"{'':4s}" + "".join(f"{s:>10s}" for s in ANCH))
for fr, fp in FRAMES.items():
    arr = load(fp)
    row = []
    for seat, (ax, ay) in ANCH.items():
        o, t = hero_rank_sig(arr, ax, ay)
        row.append(f"{o}/{t}")
    wb = wait_button(arr)
    print(f"{fr:4s}" + "".join(f"{v:>10s}" for v in row) + f"  waitBtn={wb}")
