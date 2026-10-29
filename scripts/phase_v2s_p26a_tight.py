"""P26-A — 每座 stack 文本 tight bbox 全帧测量（判稳定 ROI 常量）。
gold_rel + 列/行投影收缩 → (x0,y0,x1,y1)。列出每座每帧 bbox 与全局
min/max——用于确定是否"每座常量 ROI"足够（无 frame-id 特判）。"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))

# 宽松搜索窗（各座）
SEARCH = {
    "S1": (100, 395, 470, 505),
    "S2": (780, 285, 1120, 385),
    "S3": (1420, 400, 1800, 505),
    "S4": (60, 965, 420, 1065),
    "S5": (1480, 965, 1790, 1060),
    "S6": (780, 1215, 1110, 1325),
}


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


def tight(mask, x0, y0):
    ys, xs = np.where(mask)
    if len(xs) < 20:
        return None
    return (x0 + int(xs.min()), y0 + int(ys.min()),
            x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1)


cases = []
for ann in GT["annotations"]:
    fid = ann["frame_id"]
    st = (ann.get("fields") or {}).get("stack") or {}
    for seat, v in st.items():
        if v is not None:
            cases.append((fid, seat, v))

per_seat = defaultdict(list)
for fid, seat, value in cases:
    arr = load(fid)
    x0, y0, x1, y1 = SEARCH[seat]
    roi = arr[y0:y1, x0:x1]
    m, _ = gold_rel(roi)
    bb = tight(m, x0, y0)
    if bb is None:
        print(f"{seat} {fid[:6]} {value}: no gold")
        continue
    per_seat[seat].append((fid[:6], value, bb))
    print(f"{seat} {fid[:6]} {value}: tight=({bb[0]},{bb[1]},{bb[2]},{bb[3]})"
          f" -> w={bb[2]-bb[0]} h={bb[3]-bb[1]}")

print("\n=== 每座 ROI 常量建议（含余量）===")
for seat in sorted(per_seat):
    bbs = [b for _, _, b in per_seat[seat]]
    x_min = min(b[0] for b in bbs); x_max = max(b[2] for b in bbs)
    y_min = min(b[1] for b in bbs); y_max = max(b[3] for b in bbs)
    print(f"{seat}: tight 范围 x[{x_min},{x_max}] y[{y_min},{y_max}] "
          f"(n={len(bbs)})")
