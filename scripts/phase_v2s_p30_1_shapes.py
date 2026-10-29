"""P30-1 — 实际 observer 输出形状验证（f01 / 8bf8 / pc110 / f03）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    PositionObserver, PotObserver, StatusObserver, StackObserver,
    StreetObserver, HeroCardsObserver, BoardObserver)
from poker_decision_ai.vision2.pixel.realclient.bet_observer import (  # noqa: E402
    BetObserver)
from poker_decision_ai.vision2.pixel.realclient.runtime_snapshot import (  # noqa: E402
    build_snapshot)

EV = Path("data/vision2/coinpoker_evidence")
FRAMES = {
    "f01": "BUTTON_CHANGED_167_1787992682749_full.png",
    "8bf8": "8bf82f134651db0add626be0304a2ff1.png",
    "pc110": "POT_CHANGED_110_1787992526057_full.png",
    "f03": "POT_CHANGED_102_1787992505794_full.png",
}
for tag, fid in FRAMES.items():
    w, h, rgba = decode_png((EV / fid).read_bytes())
    observers = {
        "position_runtime": PositionObserver(),
        "pot_runtime": PotObserver(),
        "board_runtime": BoardObserver(),
        "street_runtime": StreetObserver(),
        "hero_cards_runtime": HeroCardsObserver(),
        "status_runtime": StatusObserver(),
        "stack_runtime": StackObserver(),
        "bet_runtime": BetObserver(),
    }
    for obs in observers.values():
        obs.observe(rgba, w, h)
    snap = build_snapshot(observers)
    print(f"=== {tag} ===")
    print(f"  position: {snap['position']}")
    print(f"  pot: last_pot={snap['pot']['last_pot']} status={snap['pot']['last_status']}")
    print(f"  hero_cards: {snap['hero_cards']['last_cards']} status={snap['hero_cards']['last_status']}")
    print(f"  street: {snap['street']['last_street']}")
    st = snap["status"]["last_seats"]
    print(f"  status: { {s: v.get('status') for s, v in (st or {}).items()} }")
    sk = snap["stack"]["last_seats"]
    print(f"  stack: { {s: v.get('stack') for s, v in (sk or {}).items()} }")
    bt = snap["bet"]["last_seats"]
    print(f"  bet: { {z: v.get('bet') for z, v in (bt or {}).items()} }")
    # PositionObserver full last (runtime_result 有 occupied/dealer/hero)
    pr = observers["position_runtime"].last
    rr = pr.get("runtime_result") or {}
    print(f"  rr: dealer={rr.get('dealer_seat')} hero={rr.get('hero_seat')} "
          f"uncertainty={rr.get('uncertainty')}")
    occ = rr.get("occupied_seats") or {}
    print(f"  occupied: { {s: (v.get('status') if isinstance(v, dict) else v) for s, v in occ.items()} }")
    print(f"  hero_position: {pr.get('hero_position')}")
