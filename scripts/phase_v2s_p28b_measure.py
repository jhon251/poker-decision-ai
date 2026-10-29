"""P28-B — pill 双件判据精确定量：
💬 白气泡（相对亮度，NORMAL/DIM 均处理） + 😎 gold-face（结构）。
测量各判据在 7 帧 6 座的分离，为 detector 定阈值（只测值不实施）。"""
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


def pill_zone(arr, ax, ay):
    x0, y0 = max(0, ax - 350), ay + 40
    x1, y1 = ax - 160, ay + 150
    return arr[y0:y1, x0:x1]


def white_sig(roi):
    """💬 白气泡：相对亮（l>bg×1.6）近中性。"""
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    bg = float(np.median(l))
    if bg < 2:
        return 0, bg
    m = ((l > bg * 1.6) & (np.abs(r - g) < 40) & (np.abs(g - b) < 40))
    return int(m.sum()), bg


def goldface_sig(roi):
    """😎 gold-face：相对亮 + 金 hue（r≥g，非蓝）。"""
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    bg = float(np.median(l))
    if bg < 2:
        return 0, bg
    # NORMAL gold: r>100,g>70；DIM: 相对亮。合并：l>max(60, bg*2.0)
    thr = max(60.0, bg * 2.0)
    m = ((l > thr) & (r >= g * 0.9) & (b < r + 40))
    return int(m.sum()), bg


print(f"{'':4s}" + "".join(f"{s:>14s}" for s in ANCH))
for fr, fp in FRAMES.items():
    arr = load(fp)
    row = []
    for seat, (ax, ay) in ANCH.items():
        z = pill_zone(arr, ax, ay)
        w, bg = white_sig(z)
        g, _ = goldface_sig(z)
        row.append(f"{w}/{g}(bg{bg:.0f})")
    print(f"{fr:4s}" + "".join(f"{v:>14s}" for v in row))
