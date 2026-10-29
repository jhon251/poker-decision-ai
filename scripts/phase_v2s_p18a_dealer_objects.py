"""P18-A — Dealer 视觉对象表征（READ-ONLY）。

对 6 个 GT 帧，在 GT-dealer 座 ±150px 区域内做连通域分析（红/白/暗），
判断是否存在稳定、可区分的 dealer 按钮对象（vs 仅散点像素）。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    SEAT_ANCHORS_1848)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))


def _components(arr, x0, y0, x1, y1, mask, min_area=30):
    """区域内 mask 的连通域（简单列扫描近似，返回 bbox+area 列表）。"""
    sub = mask[y0:y1, x0:x1].astype(np.uint8)
    # 用简单的行带+列带找连通域（近似）
    out = []
    ys, xs = np.where(sub > 0)
    if len(ys) < min_area:
        return out
    # 列投影分段
    cols = sub.sum(axis=0)
    active = cols >= 2
    runs, run = [], None
    for i, a in enumerate(active):
        if a and run is None:
            run = i
        elif not a and run is not None:
            runs.append((run, i))
            run = None
    if run is not None:
        runs.append((run, len(active)))
    for (cs, ce) in runs:
        seg = sub[:, cs:ce]
        rys = np.where(seg.sum(axis=1) > 0)[0]
        if len(rys) < 5:
            continue
        area = int(seg.sum())
        if area < min_area:
            continue
        out.append({"x": x0 + cs, "y": y0 + rys.min(),
                    "w": ce - cs, "h": rys.max() - rys.min() + 1,
                    "area": area})
    return out


def main() -> int:
    for ann in GT["annotations"]:
        pos = (ann.get("fields") or {}).get("position") or {}
        gt_dealer = pos.get("dealer")
        if gt_dealer is None:
            continue
        fid = ann["frame_id"]
        w, h, rgba = decode_png((EV / fid).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        ax, ay = SEAT_ANCHORS_1848[gt_dealer]
        x0, y0 = max(0, ax - 150), max(0, ay - 150)
        x1 = min(w, ax + 150)
        y1 = min(h, ay + 150)
        r = arr[..., 0].astype(np.int16)
        g = arr[..., 1].astype(np.int16)
        b = arr[..., 2].astype(np.int16)
        luma = r * 0.299 + g * 0.587 + b * 0.114
        red = (r > 150) & (g < 100) & (b < 100)
        white = luma > 200
        dark = luma < 60
        gold = (r > 120) & (g > 85) & (r >= g) & (g >= b - 30) & (luma > 90)
        print(f"\n{fid}  GT dealer={gt_dealer} anchor=({ax},{ay})")
        for label, m in [("红", red), ("白", white), ("暗", dark), ("金", gold)]:
            comps = _components(arr, x0, y0, x1, y1, m)
            if comps:
                for c in comps[:6]:
                    print(f"  {label}域: ({c['x']},{c['y']},{c['w']},{c['h']}) "
                          f"area={c['area']}")
            else:
                print(f"  {label}域: 无(min_area=30)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
