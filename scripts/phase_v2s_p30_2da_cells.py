"""P30-2D-A — 全帧候选文字区域聚类（找任何未识别的金额文本块）。
排除已知：pot capsule / seat capsules / 面板 / header / stack / hero name。
若聚类结果只剩已知对象 → 无独立 current_bet 显示。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def scan(fid, tag):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    r = arr[..., 0].astype(np.int16); g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    # 亮文本（白字/灰字——金额和标签）
    text = ((l > 150) & (np.abs(r - g) < 40) & (np.abs(g - b) < 40))
    # 网格 100x100 细胞聚类（报告每细胞文本量）
    H, W = text.shape
    cells = {}
    for cy in range(0, H, 100):
        for cx in range(0, W, 100):
            v = int(text[cy:cy+100, cx:cx+100].sum())
            if v > 400:
                cells[(cx, cy)] = v
    print(f"[{tag}] 亮文本细胞 (x,y,px):")
    for (cx, cy), v in sorted(cells.items(), key=lambda x: -x[1]):
        print(f"    ({cx},{cy}): {v}")


scan("POT_CHANGED_110_1787992526057_full.png", "pc110")
