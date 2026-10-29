"""P26-A — 两个未标注负例座格细节（f03 S6 hero / f05 S5 win-anim）。"""
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


def gold_rel(roi):
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    bg = np.median(l)
    return ((r > b + 30) & (g >= b + 18) & (r >= g * 0.9)
            & (l > bg * 1.6))


def comps_of(band2d, min_area=4):
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


S6_ROI = (838, 1236, 1040, 1308)
S5_ROI = (1560, 978, 1745, 1050)

for fid, tag, roi in (
        ("POT_CHANGED_102_1787992505794_full.png", "f03 S6 hero", S6_ROI),
        ("POT_CHANGED_139_1787992592538_full.png", "f05 S5 anim", S5_ROI)):
    arr = load(fid)
    x0, y0, x1, y1 = roi
    sub = arr[y0:y1, x0:x1]
    m = gold_rel(sub)
    comps = comps_of(m)
    print(f"[{tag}] gold_n={int(m.sum())} comps={len(comps)}")
    for c in comps:
        print(f"    comp ({c[0]},{c[1]},{c[2]},{c[3]}) area={c[4]}")
    # 背景与主要亮度
    l = sub[..., 0] * 0.299 + sub[..., 1] * 0.587 + sub[..., 2] * 0.114
    print(f"    ROI luma: bg={np.median(l):.1f} p90={np.percentile(l,90):.1f} "
          f"max={l.max():.1f}")
