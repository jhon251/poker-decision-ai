"""P29-B — 精确 c2 digit bbox：排除卡缘线（含 y<1012 行）后 split 块。"""
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


def blocks(arr, x0, y0, x1, y1, min_n=60):
    roi = arr[y0:y1, x0:x1]
    l = roi[..., 0] * 0.299 + roi[..., 1] * 0.587 + roi[..., 2] * 0.114
    ink = (l < 120).astype(np.uint8)
    h, w = ink.shape
    mask = ink.astype(bool)
    seen = np.zeros((h, w), dtype=bool)
    out = []
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
            out.append((x0 + int(min(xs)), y0 + int(min(ys2)),
                        int(max(xs) - min(xs) + 1),
                        int(max(ys2) - min(ys2) + 1), len(cells)))
    out.sort(key=lambda b: -b[4])
    return out


for fp, tag in [("8bf82f134651db0add626be0304a2ff1.png", "8bf8"),
                ("POT_CHANGED_110_1787992526057_full.png", "pc110")]:
    arr = load(fp)
    print(f"--- {tag} c2 rank wide (930-1000, 1008-1062) ---")
    for b in blocks(arr, 930, 1008, 1000, 1062):
        print(f"  block=({b[0]},{b[1]},{b[2]},{b[3]}) n={b[4]}")
    print(f"--- {tag} c2 suit wide (930-1000, 1065-1125) ---")
    for b in blocks(arr, 930, 1065, 1000, 1125):
        print(f"  block=({b[0]},{b[1]},{b[2]},{b[3]}) n={b[4]}")
