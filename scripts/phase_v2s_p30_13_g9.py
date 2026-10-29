"""P30-13 G9 — 4 真实帧回归：status/readiness/solver mode/action/provenance。"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, '.')
from poker_decision_ai.vision2_platform.png_decode import decode_png
from poker_decision_ai.vision2.pixel.realclient.production_bridge import (
    frame_to_decision_request)

EV = Path("data/vision2/coinpoker_evidence")
for fid, tag in [
        ("8bf82f134651db0add626be0304a2ff1.png", "8bf8"),
        ("BUTTON_CHANGED_167_1787992682749_full.png", "f01"),
        ("POT_CHANGED_110_1787992526057_full.png", "pc110"),
        ("POT_CHANGED_102_1787992505794_full.png", "f03")]:
    w, h, rgba = decode_png((EV / fid).read_bytes())
    res = frame_to_decision_request(rgba, w, h)
    br = res["bridge"]
    obs = br.get("observation") or {}
    print(f"[{tag}] status={res['status']} reason={res.get('reason')} "
          f"to_call={obs.get('to_call')} "
          f"hero_seat={obs.get('hero_seat')} "
          f"blinds={obs.get('blinds')} "
          f"blocking={br.get('blocking_fields')} "
          f"ah_status={obs.get('extra', {}).get('v3f_action_history_status')}")
