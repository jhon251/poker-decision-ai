"""P26-A — S3 DIM 帧加宽 ROI 重分割（423772 / 148440）。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def gold_rel(roi):
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    bg = np.median(l)
    return ((r > b + 30) & (g >= b + 18) & (r >= g * 0.9)
            & (l > bg * 1.6)), float(bg)


def comps_of(band2d, min_area=3):
    h, w = band2d.shape
    mask = band2d.astype(bool)
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
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
            if len(cells) < min_area:
                continue
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            out.append((min(xs), min(ys), max(xs) - min(xs) + 1,
                        max(ys) - min(ys) + 1, len(cells)))
    return sorted(out)


for fp, seat, x1, value in [
        ("POT_CHANGED_102_1787992505794_full.png", "S3", 1760, 423772),
        ("POT_CHANGED_13_1787992279195_full.png", "S3", 1760, 148440)]:
    arr = load(fp)
    # 宽 ROI：x 1450-1760（覆盖可能右伸），y 400-500
    for (x0, y0, xa, ya) in ((1450, 400, 1760, 500), (1560, 400, 1760, 500)):
        roi = arr[y0:ya, x0:xa]
        m, bg = gold_rel(roi)
        comps = comps_of(m)
        xs = [c[0] for c in comps]
        ws = [c[2] for c in comps]
        hs = [c[3] for c in comps]
        inner_x0 = x0 + (min(xs) if xs else 0)
        inner_x1 = x0 + (max(xs) + max([c[2] for c in comps], default=0)
                         if xs else 0)
        print(f"[{seat} {value}] ROI x{x0}-{xa}: n={len(comps)} "
              f"bbox_x={inner_x0}-{inner_x1} bg={bg:.1f} "
              f"w={','.join(str(x) for x in ws)} "
              f"h={','.join(str(x) for x in hs)}")
