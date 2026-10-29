"""P22-B — dealer 按钮结构实测（6 GT 帧）：红圆白D + 尺寸/位置。"""
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


def main() -> int:
    for ann in GT["annotations"]:
        pos = (ann.get("fields") or {}).get("position") or {}
        dealer = pos.get("dealer")
        if not isinstance(dealer, dict):
            continue
        seat = dealer.get("seat")
        bb = dealer.get("button_bbox")
        if not bb:
            continue
        fid = ann["frame_id"]
        w, h, rgba = decode_png((EV / fid).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        bx, by, bw, bh = bb
        # 按钮区域内的红/白结构
        roi = arr[by:by + bh, bx:bx + bw]
        r = roi[..., 0].astype(np.int16)
        g = roi[..., 1].astype(np.int16)
        b = roi[..., 2].astype(np.int16)
        l = r * 0.299 + g * 0.587 + b * 0.114
        red = (r > 120) & (g < 100) & (b < 100)
        white = l > 180
        red_r = (r[red].mean() if red.sum() else 0)
        white_n = int(white.sum())
        # 相对锚点偏移
        ax, ay = SEAT_ANCHORS_1848[seat]
        print(f"{fid[:32]} dealer={seat} bbox=({bx},{by},{bw},{bh}) "
              f"红={int(red.sum())}(R{red_r:.0f}) 白D={white_n} "
              f"偏移=({bx-ax},{by-ay})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
