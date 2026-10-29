"""P30-2D-A — seat capsule vs current_bet 语义对照：
pc110: S1=1,000 / S2=2,000（S2 是当前街最高下注?）+ to_call=2,000（跟注 2,000）
f01: 3 座各 21,600（S1/S2/S6 同额——多人下注后当前街 bet=21,600）
检查：最高 capsule 是否总是 == to_call（揭示 capsule=current_bet 候选）。"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    frame_to_decision_request)

EV = Path("data/vision2/coinpoker_evidence")

for fid in ["POT_CHANGED_110_1787992526057_full.png",
            "POT_CHANGED_114_1787992536082_full.png",
            "BUTTON_CHANGED_167_1787992682749_full.png"]:
    w, h, rgba = decode_png((EV / fid).read_bytes())
    res = frame_to_decision_request(rgba, w, h)
    obs = res["bridge"].get("observation") or {}
    bets = obs.get("extra", {}).get("v3f_bets", {})
    print(f"{fid[:28]}:")
    print(f"  v3f_bets={bets}")
    print(f"  to_call={obs.get('to_call')}")
    vals = [v for v in bets.values() if v is not None]
    if vals:
        print(f"  max_seat_bet={max(vals)}")
