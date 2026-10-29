"""P30-2B — bridge 集成冒烟：真实帧 blinds/ante 流转。"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    frame_to_decision_request)

EV = Path("data/vision2/coinpoker_evidence")
w, h, rgba = decode_png((EV / "8bf82f134651db0add626be0304a2ff1.png")
                        .read_bytes())
res = frame_to_decision_request(rgba, w, h)
br = res["bridge"]
obs = br.get("observation") or {}
print(f"status={res['status']}")
print(f"  blinds={obs.get('blinds')} ante={obs.get('ante')}")
print(f"  gaps={br['decision_gaps']}")
print(f"  blinds_conf={br['field_confidences'].get('blinds')} "
      f"ante_conf={br['field_confidences'].get('ante')}")
print(f"  'blinds' in gaps: {'blinds' in br['decision_gaps']}")
