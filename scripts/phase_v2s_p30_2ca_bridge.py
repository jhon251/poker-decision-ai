"""P30-2C-A — bridge 集成验证：真实帧 to_call 流转。"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    frame_to_decision_request)

EV = Path("data/vision2/coinpoker_evidence")
for fid in ["POT_CHANGED_114_1787992536082_full.png",
            "POT_CHANGED_110_1787992526057_full.png"]:
    w, h, rgba = decode_png((EV / fid).read_bytes())
    res = frame_to_decision_request(rgba, w, h)
    br = res["bridge"]
    obs = br.get("observation") or {}
    print(f"{fid[:28]}: status={res['status']} "
          f"to_call={obs.get('to_call')} "
          f"gaps={'to_call' in br['decision_gaps']}")
