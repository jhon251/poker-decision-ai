"""P22-B — 最终检测方案测试：低红阈值 + 紧凑 + 排除board区 + 座关联。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.geometry import connected_components  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    SEAT_ANCHORS_1848)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))


def _button_cands(arr):
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    # 低红阈值（按钮红 r60-160 均捕获）+ 高红绿比
    red = ((r > 40) & (g < 60) & (b < 60) & (r > g * 2)).astype(np.uint8)
    # 排除 board 区（红牌花色）与 hero 区
    red[480:745, :] = 0
    red[1000:1230, :] = 0
    cands = []
    for c in connected_components(red, min_area=80):
        w, h = int(c.bbox.width), int(c.bbox.height)
        if not (14 <= w <= 70 and 14 <= h <= 70):
            continue
        ar = w / max(1.0, h)
        if not (0.4 <= ar <= 1.8):
            continue
        x, y = int(c.bbox.x), int(c.bbox.y)
        cands.append({"bbox": [x, y, w, h], "area": c.area})
    return cands


def main() -> int:
    ok = 0
    total = 0
    misses = []
    for ann in GT["annotations"]:
        pos = (ann.get("fields") or {}).get("position") or {}
        dealer = pos.get("dealer")
        if not isinstance(dealer, dict):
            continue
        gt_seat = dealer.get("seat")
        total += 1
        fid = ann["frame_id"]
        w, h, rgba = decode_png((EV / fid).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        cands = _button_cands(arr)
        best = None
        for c in cands:
            bx, by, bw, bh = c["bbox"]
            for seat, (ax, ay) in SEAT_ANCHORS_1848.items():
                d = float(((bx + bw // 2 - ax) ** 2
                           + (by + bh // 2 - ay) ** 2) ** 0.5)
                if d <= 400 and (best is None or d < best[1]):
                    best = ((bx, by, bw, bh), d, seat)
        pred = best[2] if best else None
        hit = pred == gt_seat
        if hit:
            ok += 1
        else:
            misses.append((fid[:30], gt_seat, pred, best[0] if best else None,
                           len(cands)))
        print(f"{fid[:30]} GT={gt_seat} pred={pred} 候选={len(cands)} "
              f"{'HIT' if hit else 'MISS'}")
    print(f"\nOK: {ok}/{total}")
    for m in misses:
        print("  MISS:", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
