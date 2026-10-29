"""P18 — 红 D 对象特征实测（READ-ONLY）：每个 GT 帧 GT-dealer 座附近红 blob 的 bbox/面积/形状。"""
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


def _red_blob(arr, ax, ay, radius=120):
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    red = ((r > 150) & (g < 100) & (b < 100))
    x0, y0 = max(0, ax - radius), max(0, ay - radius)
    x1 = min(arr.shape[1], ax + radius)
    y1 = min(arr.shape[0], ay + radius)
    sub = red[y0:y1, x0:x1]
    ys, xs = np.where(sub)
    if len(ys) < 10:
        return None
    return (x0 + xs.min(), y0 + ys.min(), xs.max() - xs.min() + 1,
            ys.max() - ys.min() + 1, len(ys))


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
        bl = _red_blob(arr, ax, ay)
        print(f"{fid}: GT dealer={gt_dealer} anchor=({ax},{ay}) "
              f"红blob={bl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
