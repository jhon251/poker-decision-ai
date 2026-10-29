"""P19-A — Bet 1876 PRE-AUDIT：候选 per-seat ROI 在 4 个 GT 帧上的读取对照。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.coinpoker_bet import (  # noqa: E402
    gray_luma, recognize_bet_capsule, load_bet_templates)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))
tpls = load_bet_templates()

#: 候选 per-seat ROI（朝桌心方向，容纳 bet 胶囊）
ROIS_1876 = {
    "S1": (250, 370, 500, 180),
    "S2": (650, 270, 450, 190),
    "S3": (1300, 270, 420, 190),
    "S4": (80, 900, 320, 170),
    "S5": (1400, 740, 430, 330),
    "S6": (740, 750, 360, 140),
}


def main() -> int:
    for ann in GT["annotations"]:
        bet = (ann.get("fields") or {}).get("bet") or {}
        if not bet:
            continue
        fid = ann["frame_id"]
        w, h, rgba = decode_png((EV / fid).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        lum = gray_luma(arr)
        print(f"\n{fid}  GT bet={bet}")
        for seat, gt_val in bet.items():
            roi = ROIS_1876[seat]
            res = recognize_bet_capsule(lum, roi, tpls)
            pred = res.get("bet")
            print(f"  {seat}: GT={gt_val}  pred={pred} "
                  f"status={res['status']} raw={res.get('raw_text')} "
                  f"reason={res.get('reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
