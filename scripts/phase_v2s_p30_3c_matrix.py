"""P30-3-C — Decision Readiness 矩阵：真实 1876 帧全字段（V3F→bridge）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.runtime_snapshot import (  # noqa: E402
    build_snapshot)
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    PositionObserver, PotObserver, BoardObserver, StreetObserver,
    HeroCardsObserver, StatusObserver, StackObserver)
from poker_decision_ai.vision2.pixel.realclient.bet_observer import (  # noqa: E402
    BetObserver)
from poker_decision_ai.vision2.pixel.realclient.betting_header_observer import (  # noqa: E402
    BettingHeaderObserver)
from poker_decision_ai.vision2.pixel.realclient.to_call_observer import (  # noqa: E402
    ToCallObserver)
from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    snapshot_to_vision_observation)

EV = Path("data/vision2/coinpoker_evidence")

FRAMES = ["8bf82f134651db0add626be0304a2ff1.png",
          "POT_CHANGED_110_1787992526057_full.png",
          "POT_CHANGED_114_1787992536082_full.png",
          "BUTTON_CHANGED_167_1787992682749_full.png"]


def run_frame(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    observers = {
        "position_runtime": PositionObserver(),
        "pot_runtime": PotObserver(),
        "board_runtime": BoardObserver(),
        "street_runtime": StreetObserver(),
        "hero_cards_runtime": HeroCardsObserver(),
        "status_runtime": StatusObserver(),
        "stack_runtime": StackObserver(),
        "bet_runtime": BetObserver(),
        "betting_header_runtime": BettingHeaderObserver(),
        "to_call_runtime": ToCallObserver(),
    }
    for obs in observers.values():
        obs.observe(arr, w, h)
    snap = build_snapshot(observers)
    snap["blind"] = observers["betting_header_runtime"].summary()
    snap["to_call"] = observers["to_call_runtime"].summary()
    pr = (observers["position_runtime"].last or {}).get("runtime_result") or {}
    br = snapshot_to_vision_observation(snap, pr)
    return snap, br


print("=== 字段矩阵（8bf8）===")
snap, br = run_frame(FRAMES[0])
obs = br["observation"] or {}
print(f"bridge: {br['status']} reason={br['reason']}")
print(f"uncertain: {br.get('warnings') or []}")
fld = {
    "pot": (snap.get("pot") or {}).get("last_status"),
    "street": (snap.get("street") or {}).get("last_street"),
    "board": (snap.get("board") or {}).get("last_status"),
    "hero_cards": (snap.get("hero_cards") or {}).get("last_status"),
    "stack": (snap.get("stack") or {}).get("last_status"),
    "bet": (snap.get("bet") or {}).get("last_status"),
    "status": (snap.get("status") or {}).get("last_status"),
    "position_unc": (snap.get("position") or {}).get("last_uncertainty"),
    "blind": (snap.get("blind") or {}).get("last_status"),
    "to_call": (snap.get("to_call") or {}).get("last_status"),
}
for k, v in fld.items():
    print(f"  {k}: V3F={v}")
print(f"  bridge confidences: {br.get('field_confidences')}")
