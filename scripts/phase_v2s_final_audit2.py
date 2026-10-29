"""FINAL AUDIT — f03 hero/occupancy 独立性。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.hero_seat import (  # noqa: E402
    HeroSeatDetector1876)
from poker_decision_ai.vision2.pixel.realclient.seat_occupancy import (  # noqa: E402
    SeatOccupancyRecognizer1876)

EV = Path("data/vision2/coinpoker_evidence")
w, h, rgba = decode_png((EV / "POT_CHANGED_102_1787992505794_full.png")
                        .read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
hres = HeroSeatDetector1876().detect(arr)
print(f"hero (f03): {hres['hero_seat']} ({hres['reason']})")
hero_ev = hres["evidence"]["S6"]
print(f"  S6 hero evidence: {hero_ev}")
occ = SeatOccupancyRecognizer1876().recognize(arr)
print(f"S6 occupancy (f03): {occ['S6']['status']} ({occ['S6']['reason']})")
print(f"  S6 evidence: {occ['S6']['evidence']}")
# 关键：hero 由 own-seat pill 判定（不受座牌盒遮挡影响）；occupancy 由
# 座牌盒判定（hero 卡 overlay 遮挡 → UNKNOWN）——两独立证据源，无互相推导。
