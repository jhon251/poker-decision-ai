"""FINAL AUDIT — table_topology / hero_position 完整链（f01 全可见帧）。"""
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
print(f"table_topology: {rr.get('table_topology')}")
print(f"hero_seat: {rr.get('hero_seat')}  dealer_seat: {rr.get('dealer_seat')}")
print(f"uncertainty: {rr.get('uncertainty')}")
print(f"hero_position: {out.get('hero_position')}")
print(f"positions_ring: {out.get('positions_ring')}")
occ = rr.get("occupied_seats", {})
print(f"occupied: {[(s, v.get('status') if isinstance(v, dict) else v) for s, v in occ.items()]}")
