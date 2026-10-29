"""P30-3-C — 深挖 status/bet/stack 的 V3F summary 实际值。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    StatusObserver, StackObserver)
from poker_decision_ai.vision2.pixel.realclient.bet_observer import (  # noqa: E402
    BetObserver)

EV = Path("data/vision2/coinpoker_evidence")
w, h, rgba = decode_png((EV /
    "8bf82f134651db0add626be0304a2ff1.png").read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)

so = StatusObserver()
r1 = so.observe(arr, w, h)
print("StatusObserver.observe →")
print("  last_status:", (so.last or {}).get("last_status"))
print("  last_seats:", (so.last or {}).get("last_seats"))
print("  summary:", so.summary())

sto = StackObserver()
r2 = sto.observe(arr, w, h)
print("\nStackObserver →")
print("  last:", {k: v for k, v in (sto.last or {}).items()
                 if k != "last_seats"})
print("  last_seats:", (sto.last or {}).get("last_seats"))

bo = BetObserver()
r3 = bo.observe(arr, w, h)
print("\nBetObserver →")
print("  last:", {k: v for k, v in (bo.last or {}).items()
                 if k != "last_seats"})
print("  last_seats:", (bo.last or {}).get("last_seats"))
