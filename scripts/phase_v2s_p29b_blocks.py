"""P29-B — 逐帧逐卡角标 tight bbox（白面相对定位）。
方法：对每个窗口，从"白面内"提取最大墨连通块（排除卡缘/背景）。
"""
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


def biggest_ink_block(arr, x0, y0, x1, y1, tag):
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    ink = (l < 120).astype(np.uint8)
    # 最大连通
    h, w = ink.shape
    mask = ink.astype(bool)
    seen = np.zeros((h, w), dtype=bool)
    best = None
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]; seen[sy, sx] = True
            cells = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cx, cy))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] \
                                and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            if len(cells) < 25:
                continue
            xs = [c[0] for c in cells]; ys2 = [c[1] for c in cells]
            if best is None or len(cells) > best[4]:
                best = (min(xs), min(ys2), max(xs) - min(xs) + 1,
                        max(ys2) - min(ys2) + 1, len(cells))
    if best is None:
        print(f"[{tag}] no block")
        return
    red = (r > 110) & (r > g + 30) & (r > b + 30)
    print(f"[{tag}] block=({x0+best[0]},{y0+best[1]},{best[2]},{best[3]}) "
          f"n={best[4]} red_frac={float(red[ink.astype(bool)].mean()):.2f}")


for fp, c1, c2 in [
        ("8bf82f134651db0add626be0304a2ff1.png", "5c", "3c"),
        ("POT_CHANGED_110_1787992526057_full.png", "6c", "3h"),
        ("POT_CHANGED_102_1787992505794_full.png", "9c", "3d")]:
    arr = load(fp)
    print(f"--- {fp[:28]} GT={c1}{c2} ---")
    biggest_ink_block(arr, 815, 1012, 875, 1066, "c1_rank")
    biggest_ink_block(arr, 815, 1066, 878, 1120, "c1_suit")
    biggest_ink_block(arr, 833, 1008, 895, 1065, "c2_rank")
    biggest_ink_block(arr, 833, 1065, 900, 1122, "c2_suit")
