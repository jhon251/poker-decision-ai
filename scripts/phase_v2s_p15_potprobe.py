"""P15 — pot 1876 精确定位 + 数字尺度测量（只读，快速）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402


def main() -> int:
    frames = sorted(Path("data/vision2/coinpoker_evidence").glob("*_full.png"))
    shown = 0
    for p in frames:
        w, h, rgba = decode_png(p.read_bytes())
        if (w, h) != (1876, 1400):
            continue
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        # 顶中宽区 700-1150, 60-180
        roi = arr[60:180, 700:1150]
        r = roi[..., 0].astype(np.int16)
        g = roi[..., 1].astype(np.int16)
        b = roi[..., 2].astype(np.int16)
        luma = r * 0.299 + g * 0.587 + b * 0.114
        # gold mask (r>120,g>85, r>=g)
        gold = (r > 120) & (g > 85) & (r >= g)
        # white mask
        white = luma > 150
        if gold.sum() < 200:
            continue
        # 找到 gold 文本的 bbox
        ys, xs = np.where(gold)
        y0, y1 = ys.min() + 60, ys.max() + 60
        x0, x1 = xs.min() + 700, xs.max() + 700
        # 逐行/逐列投影，找数字串（排除中文标签——中文标签更宽更矮？）
        # 报告数字尺度：gold 行的列投影
        rows = gold.sum(axis=1)
        band_rows = np.where(rows > 5)[0]
        if len(band_rows) == 0:
            continue
        print(f"{p.name}: gold bbox=({x0},{y0},{x1-x0+1},{y1-y0+1}) "
              f"rows={len(band_rows)} 白像素={int(white.sum())} "
              f"gold={int(gold.sum())}")
        # 每行 gold 宽度（找连续段）
        shown += 1
        if shown >= 8:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
