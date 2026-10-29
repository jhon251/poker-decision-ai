"""P28-A — own-seat 专属件定量矩阵：
A) emoji pill 簇（黄 😎：r>150,g>100,b<120）在 座锚 x-345..-165, y+35..+150
B) 手牌力条（rank bar：橙黄条）在 座锚 x-120..+120, y+165..+235
对 6 GT 帧 + 用户帧 全座扫描：独占性（仅 hero 座出现）。"""
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


def emoji_yellow(arr, ax, ay):
    x0, y0 = max(0, ax - 345), ay + 35
    x1, y1 = ax - 165, ay + 150
    roi = arr[y0:y1, x0:x1]
    if roi.size == 0:
        return 0
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    return int(((r > 150) & (g > 100) & (b < 120) & (r >= g)).sum())


def rankbar_sig(arr, ax, ay):
    """座锚下方 y+165..+235 的橙黄条（进度条 rail + 填充）。"""
    x0, y0 = max(0, ax - 130), ay + 160
    x1, y1 = ax + 130, ay + 240
    roi = arr[y0:y1, x0:x1]
    if roi.size == 0:
        return 0, 0
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    orange = ((r > 140) & (g > 80) & (b < 100)).sum()
    l = r * 0.299 + g * 0.587 + b * 0.114
    gray_rail = ((l > 90) & (l < 180) & (np.abs(r - g) < 30)).sum()
    return int(orange), int(gray_rail)


print(f"{'':4s} {'seat':4s} {'emojiY':>7s} {'rankOr':>7s} {'rail':>6s}")
for fr, fp in FRAMES.items():
    arr = load(fp)
    for seat, (ax, ay) in ANCH.items():
        e = emoji_yellow(arr, ax, ay)
        o, rail = rankbar_sig(arr, ax, ay)
        mark = " <-- HERO?" if (e > 200 or o > 200) else ""
        print(f"{fr:4s} {seat:4s} {e:7d} {o:7d} {rail:6d}{mark}")
