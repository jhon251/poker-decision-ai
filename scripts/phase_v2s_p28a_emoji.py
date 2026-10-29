"""P28-A — emoji 反应 pill（💬😎）是否为 own-seat 专属 UI：
扫描 f01/f02/f03/f04/f05/f06 + 用户帧 每座左侧区域，统计黄色 emoji pill。
"""
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


def emoji_sig(arr, seat):
    """座牌盒左侧（座锚点左下 200px 区）黄橙 emoji pill。"""
    ax, ay = ANCH[seat]
    # 座牌盒左缘 ≈ ax-100；emoji 在其左 90px
    x0, y0 = max(0, ax - 220), min(arr.shape[0] - 1, ay + 30)
    x1 = max(0, ax - 100)
    y1 = min(arr.shape[0], ay + 180)
    roi = arr[y0:y1, x0:x1]
    if roi.size == 0:
        return 0, 0.0
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    # 黄橙（emoji 😎 底色）+ 中性白（💬 pill）
    yellow = ((r > 150) & (g > 110) & (b < 110) & (r >= g)).sum()
    l = r * 0.299 + g * 0.587 + b * 0.114
    white = ((l > 200) & (np.abs(r - g) < 25) & (np.abs(g - b) < 30)).sum()
    return int(yellow), float(white)


print(f"{'':4s}" + "".join(f"{s:>10s}" for s in ANCH))
for fr, fp in FRAMES.items():
    arr = load(fp)
    row = []
    for seat, (ax, ay) in ANCH.items():
        y, w = emoji_sig(arr, seat)
        row.append(f"{y:4d}/{int(w):4d}")
    print(f"{fr:4s}" + "".join(f"{v:>10s}" for v in row))
