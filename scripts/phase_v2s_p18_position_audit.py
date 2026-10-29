"""P18 — Position 1876 READ-ONLY PRE-AUDIT：6 个 GT 帧的 dealer/occupied/hero 实测。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    SeatOccupancyDetector, HeroSeatDetector, DealerSeatMapper,
    SEAT_ANCHORS_1848, window_anchors)
from poker_decision_ai.vision2.pixel.realclient.cashgame import (  # noqa: E402
    DealerRecognizer)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))


def main() -> int:
    for ann in GT["annotations"]:
        pos = (ann.get("fields") or {}).get("position") or {}
        gt_dealer = pos.get("dealer")
        if gt_dealer is None:
            continue
        fid = ann["frame_id"]
        w, h, rgba = decode_png((EV / fid).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        # 红 D 检测
        dres = DealerRecognizer().recognize(arr)
        dmap = DealerSeatMapper().map(dres)
        pred_dealer = dmap["dealer_seat"]
        # 红 D 中心
        r = arr[..., 0].astype(np.int16)
        g = arr[..., 1].astype(np.int16)
        b = arr[..., 2].astype(np.int16)
        red = (r > 150) & (g < 100) & (b < 100)
        ys, xs = np.where(red)
        red_n = len(xs)
        red_center = (int(xs.mean()), int(ys.mean())) if red_n else None
        # occupied（bright gold blobs）
        occ = SeatOccupancyDetector(window="1848").detect(arr)
        occ_states = {s: occ[s]["status"] for s in occ}
        # hero
        hero = HeroSeatDetector(window="1848").detect(arr)["hero_seat"]
        print(f"\n{fid}")
        print(f"  GT dealer={gt_dealer}  pred={pred_dealer}  "
              f"{'OK' if pred_dealer == gt_dealer else 'FAIL'}")
        print(f"  红D: n={red_n} center={red_center} "
              f"bbox={dres.get('bbox')}")
        # 红 D 中心到 1848 锚点距离
        if red_center:
            dists = {s: round(float(((red_center[0]-ax)**2 + (red_center[1]-ay)**2)**0.5))
                     for s, (ax, ay) in SEAT_ANCHORS_1848.items()}
            print(f"  红D 到 1848 锚点距离: {dists}")
        print(f"  occupied(1848): {occ_states}")
        print(f"  hero: {hero} (GT 约定 S6)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
