"""P24 — Bet 1876 PRE-AUDIT：图像测量 ROI + 胶囊识别实测。"""
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
print("v2bet1280 模板:", sorted(tpls.keys()))

#: per-seat 候选 ROI（P16-E 实测 + P21 图像定位，(x0,y0,宽,高)）
ROIS = {
    "S1": (440, 455, 120, 40),
    "S2": (680, 390, 120, 40),
    "S6": (815, 785, 115, 40),
    "S5": (1320, 890, 120, 45),
}


def main() -> int:
    ok = 0
    tot = 0
    for ann in GT["annotations"]:
        bet = (ann.get("fields") or {}).get("bet") or {}
        if not bet:
            continue
        fid = ann["frame_id"]
        w, h, rgba = decode_png((EV / fid).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        lum = gray_luma(arr)
        for seat, gt_val in bet.items():
            tot += 1
            res = recognize_bet_capsule(lum, ROIS[seat], tpls)
            pred = res.get("bet")
            hit = (pred == gt_val)
            if hit:
                ok += 1
            print(f"{fid[:26]} {seat} GT={gt_val} pred={pred} "
                  f"status={res['status']} reason={res.get('reason')} "
                  f"{'HIT' if hit else 'MISS'}")
    print(f"\nOK: {ok}/{tot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
