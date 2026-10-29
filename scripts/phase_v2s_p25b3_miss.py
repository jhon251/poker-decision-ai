"""P25-B.3 — 检查漏检正例：按已知座偏移采样期望 tag 位置的像素颜色。"""
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


FRAMES = {
    "f01": "BUTTON_CHANGED_167_1787992682749_full.png",
    "f02": "POT_CHANGED_12_1787995290794_full.png",
    "f03": "POT_CHANGED_102_1787992505794_full.png",
    "f04": "POT_CHANGED_13_1787992279195_full.png",
    "f05": "POT_CHANGED_139_1787992592538_full.png",
    "f06": "BUTTON_CHANGED_598_1787993753587_full.png",
}
# 已确认 tag 原位（每座固定 bbox）
TAG = {"S2": (816, 211, 103, 40), "S3": (1494, 325, 103, 41),
       "S4": (101, 891, 103, 40), "S5": (1531, 891, 103, 40),
       "S6": (816, 1149, 103, 40)}
MISS = [("f02", "S4"), ("f03", "S1"), ("f03", "S3"), ("f03", "S4"),
        ("f03", "S5"), ("f04", "S3"), ("f04", "S4"), ("f04", "S5"),
        ("f04", "S6")]

for fr, seat in MISS:
    if seat not in TAG:
        print(f"{fr} {seat}: no expected anchor")
        continue
    arr = load(FRAMES[fr])
    bx, by, bw, bh = TAG[seat]
    roi = arr[by:by + bh, bx:bx + bw]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    # 中心 5x5 采样
    cy, cx = bh // 2, bw // 2
    c = roi[cy - 2:cy + 3, cx - 2:cx + 3].reshape(-1, 4)
    cm = c[:, :3].mean(axis=0)
    print(f"{fr} {seat}: center rgb=({cm[0]:.0f},{cm[1]:.0f},{cm[2]:.0f}) "
          f"luma_mean={l.mean():.0f} luma_p50={np.percentile(l,50):.0f} "
          f"luma_p90={np.percentile(l,90):.0f} "
          f"dark_frac={float((l<32).mean()):.2f}")
