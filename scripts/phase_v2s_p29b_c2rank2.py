"""P29-B — c2 角标：避开卡缘线（y>1012）后找 digit + suit 块。"""
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


def biggest(arr, x0, y0, x1, y1, tag, min_n=25):
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    ink = (l < 120).astype(np.uint8)
    h, w = ink.shape
    mask = ink.astype(bool)
    seen = np.zeros((h, w), dtype=bool)
    blocks = []
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
            if len(cells) < min_n:
                continue
            xs = [c[0] for c in cells]; ys2 = [c[1] for c in cells]
            blocks.append((x0 + min(xs), y0 + min(ys2),
                           max(xs) - min(xs) + 1,
                           max(ys2) - min(ys2) + 1, len(cells)))
    blocks.sort(key=lambda b: -b[4])
    for b in blocks[:3]:
        print(f"[{tag}] block=({b[0]},{b[1]},{b[2]},{b[3]}) n={b[4]}")


for fp, tag in [("8bf82f134651db0add626be0304a2ff1.png", "8bf8"),
                ("POT_CHANGED_110_1787992526057_full.png", "pc110")]:
    arr = load(fp)
    # c2 rank：y 1014-1065（避 y<1014 卡缘线），x 831-900
    biggest(arr, 831, 1014, 900, 1065, f"{tag} c2_rank")
    biggest(arr, 831, 1065, 900, 1125, f"{tag} c2_suit")
