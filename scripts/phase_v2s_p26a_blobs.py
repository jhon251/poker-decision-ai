"""P26-A — 每 GT 座区域全部 golden blob 定位（无 P23 bbox 假设）。

座区域 = 锚点 ±250/±220（覆盖座牌盒与桌沿）；gold 掩码连通组件按面积排序；
报告每个 blob 的 bbox/area/宽高——用于确立真实 stack 文本几何。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))

ANCH = {"S1": (255, 310), "S2": (918, 210), "S3": (1578, 320),
        "S4": (218, 868), "S5": (1618, 868), "S6": (945, 1160)}


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def gold_mask(frame):
    r = frame[..., 0].astype(np.int16); g = frame[..., 1].astype(np.int16)
    b = frame[..., 2].astype(np.int16)
    return ((r > 100) & (g > 70) & (r >= g)
            & (g >= b + 25) & (r > b + 40)).astype(np.uint8)


def blobs(arr, x0, y0, x1, y1, min_area=20):
    sub = gold_mask(arr[y0:y1, x0:x1])
    h, w = sub.shape
    mask = sub.astype(bool)
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
            if len(cells) < min_area:
                continue
            xs = [c[0] for c in cells]; ys = [c[1] for c in cells]
            out.append((len(cells), x0 + min(xs), y0 + min(ys),
                        max(xs) - min(xs) + 1, max(ys) - min(ys) + 1))
    return sorted(out, reverse=True)


cases = []
for ann in GT["annotations"]:
    fid = ann["frame_id"]
    st = (ann.get("fields") or {}).get("stack") or {}
    for seat, v in st.items():
        if v is not None:
            cases.append((fid, seat, v))

print(f"{'seat':4s} {'value':>9s} {'fid':7s}  blobs(area, x,y,w,h)")
for fid, seat, value in cases:
    arr = load(fid)
    ax, ay = ANCH[seat]
    x0, y0 = max(0, ax - 250), max(0, ay - 220)
    x1, y1 = min(arr.shape[1], ax + 250), min(arr.shape[0], ay + 220)
    bs = blobs(arr, x0, y0, x1, y1)
    top = bs[:6]
    print(f"{seat:4s} {value:>9d} {fid[:7]:7s}  "
          + " | ".join(f"({a},{x},{y},{w},{h})" for a, x, y, w, h in top))
