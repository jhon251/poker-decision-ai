"""FINAL AUDIT — f01 position UNKNOWN 原因诊断（只读检查）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    PositionObserver)

EV = Path("data/vision2/coinpoker_evidence")
w, h, rgba = decode_png((EV / "BUTTON_CHANGED_167_1787992682749_full.png")
                        .read_bytes())
out = PositionObserver().observe(rgba, w, h)
rr = out["runtime_result"]
print("uncertainty:", rr.get("uncertainty"))
print("reason:", rr.get("reason") or out.get("reason"))
# 深入：谁能返回 UNKNOWN？
# compute_positions: dealer_or_hero_missing / occupied_incomplete_* /
# no_occupied_seats / illegal_* / dealer_not_occupied / hero_not_occupied
print("\nfull runtime_result keys:", list(rr.keys()))
# 手动重放 compute_positions 逻辑
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    ring_number)
occ = rr["occupied_seats"]
occ_status = {s: (v.get("status") if isinstance(v, dict) else v)
              for s, v in occ.items()}
print("\noccupied status map:", occ_status)
hero = rr.get("hero_seat")
dealer = rr.get("dealer_seat")
print(f"hero={hero} dealer={dealer}")
print(f"hero in occupied: {hero in occ_status and occ_status[hero] == 'OCCUPIED'}")
print(f"dealer in occupied: {dealer in occ_status and occ_status[dealer] == 'OCCUPIED'}")
print(f"hero ring: {ring_number(hero) if hero else None}")
print(f"dealer ring: {ring_number(dealer) if dealer else None}")
