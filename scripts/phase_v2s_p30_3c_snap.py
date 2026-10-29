"""P30-3-C — 深挖：bridge 到底读到什么（stack/bet/status summary 内容）。"""
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

EV = Path("data/vision2/coinpoker_evidence")
w, h, rgba = decode_png((EV /
    "8bf82f134651db0add626be0304a2ff1.png").read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)

obs = {
    "position_runtime": PositionObserver(),
    "pot_runtime": PotObserver(),
    "board_runtime": BoardObserver(),
    "street_runtime": StreetObserver(),
    "hero_cards_runtime": HeroCardsObserver(),
    "status_runtime": StatusObserver(),
    "stack_runtime": StackObserver(),
    "bet_runtime": BetObserver(),
}
for o in obs.values():
    o.observe(arr, w, h)
snap = build_snapshot(obs)
print("snapshot keys:", list(snap.keys()))
print("\nstack summary:", snap["stack"])
print("\nbet summary:", snap["bet"])
print("\nstatus summary keys:", list(snap["status"].keys()))
print("status last_seats:", snap["status"].get("last_seats"))
