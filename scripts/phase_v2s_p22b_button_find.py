"""P22-B — dealer 按钮全帧搜索（紧凑红 blob + 白D 验证），6 帧对照 GT。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    SEAT_ANCHORS_1848)
from poker_decision_ai.vision2.pixel.geometry import connected_components  # noqa: E402
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))


def _find_button(arr):
    """全帧找 dealer 按钮：紧凑红 blob(18-50px, 近方形) + 中心白D，
    排除 board 区(y500-750)与 hero 区(y1000-1200)的红牌花色。"""
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    red = ((r > 120) & (g < 100) & (b < 100)).astype(np.uint8)
    # 排除 board/hero 牌花色区
    red[500:760, :] = 0
    red[1000:1220, :] = 0
    cands = []
    for c in connected_components(red, min_area=60):
        w = int(c.bbox.width)
        h = int(c.bbox.height)
        if not (16 <= w <= 65 and 16 <= h <= 65):
            continue
        ar = w / max(1.0, h)
        if not (0.5 <= ar <= 1.6):
            continue             # 紧凑（非高窄牌花色）
        cx, cy = int(c.bbox.x), int(c.bbox.y)
        # 白 D：blob 区域白像素占比（按钮=D字母+红环；牌背白为边界，占比低）
        sub_l = l[cy:cy + h, cx:cx + w]
        white_n = int((sub_l > 180).sum())
        if white_n < 10 or white_n / (w * h) < 0.15:
            continue
        cands.append({"bbox": [cx, cy, w, h], "white_n": white_n})
    return cands


def main() -> int:
    ok = 0
    total = 0
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
        cands = _find_button(arr)
        # 每个候选关联最近锚点
        best = None
        for c in cands:
            bx, by, bw, bh = c["bbox"]
            for seat, (ax, ay) in SEAT_ANCHORS_1848.items():
                d = float(((bx + bw // 2 - ax) ** 2
                           + (by + bh // 2 - ay) ** 2) ** 0.5)
                if d <= 350 and (best is None or d < best[1]):
                    best = ((bx, by, bw, bh), d, seat)
        pred = best[2] if best else None
        hit = (pred == gt_seat)
        if hit:
            ok += 1
        print(f"{fid[:30]} GT={gt_seat} pred={pred} 候选={len(cands)} "
              f"{'HIT' if hit else 'MISS'}" + (f" bbox={best[0]}" if best else ""))
    print(f"\nOK: {ok}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
