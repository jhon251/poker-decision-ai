"""P30-1 — bridge 冒烟测试：真实帧 + UNKNOWN 合成。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    snapshot_to_vision_observation, frame_to_decision_request)

EV = Path("data/vision2/coinpoker_evidence")

# 1) 真实帧
for fid in ["BUTTON_CHANGED_167_1787992682749_full.png",
            "8bf82f134651db0add626be0304a2ff1.png",
            "POT_CHANGED_110_1787992526057_full.png",
            "POT_CHANGED_102_1787992505794_full.png"]:
    w, h, rgba = decode_png((EV / fid).read_bytes())
    res = frame_to_decision_request(rgba, w, h)
    br = res["bridge"]
    obs = br.get("observation") or {}
    print(f"{fid[:28]}: status={res['status']} "
          f"hero={obs.get('hero_seat')} button={obs.get('button_seat')} "
          f"pot={obs.get('pot')} cards={obs.get('hole_cards')} "
          f"seats={len(obs.get('seats') or [])} "
          f"v3f_hp={obs.get('extra', {}).get('v3f_hero_position')}")
    if res["status"] != "OK":
        print(f"    reason={res['reason']}")

# 2) 纯桥接：全已知合成 snapshot → OK
snap_ok = {
    "street": {"last_street": "PREFLOP"},
    "pot": {"last_pot": 8920, "last_status": "RECOGNIZED"},
    "hero_cards": {"last_cards": ["5c", "3c"], "last_status": "RECOGNIZED"},
    "board": {"last_cards": []},
    "bet": {"last_seats": {}},
    "status": {"last_seats": {f"S{i}": {"status": "NOT_OBSERVED"}
                              for i in range(1, 7)}},
    "stack": {"last_seats": {f"S{i}": {"stack": 1000 * i,
                                       "status": "RECOGNIZED"}
                             for i in range(1, 7)}},
    "position": {"last_hero_position": "CO", "last_uncertainty":
                 "RECOGNIZED"},
}
pr_ok = {"occupied_seats": {f"S{i}": {"status": "OCCUPIED"}
                            for i in range(1, 7)},
         "dealer_seat": "S4", "hero_seat": "S6", "table_topology": "6-max"}
br = snapshot_to_vision_observation(snap_ok, pr_ok)
print(f"\n合成全已知: status={br['status']}")
print(json_ := __import__("json").dumps(br["observation"], indent=1,
                                        ensure_ascii=False)[:900])

# 3) pot UNKNOWN → 非 0
snap_unk = dict(snap_ok)
snap_unk["pot"] = {"last_pot": None, "last_status": "UNKNOWN"}
br2 = snapshot_to_vision_observation(snap_unk, pr_ok)
print(f"\npot UNKNOWN: status={br2['status']} "
      f"pot={br2['observation'].get('pot')} (必须 None 非 0)")
assert br2["observation"]["pot"] is None

# 4) occupied UNKNOWN → 拒绝
pr_unk = dict(pr_ok)
pr_unk["occupied_seats"] = {f"S{i}": {"status": "OCCUPIED"}
                            for i in range(1, 7)}
pr_unk["occupied_seats"]["S6"] = {"status": "UNKNOWN"}
br3 = snapshot_to_vision_observation(snap_ok, pr_unk)
print(f"\noccupied S6 UNKNOWN: status={br3['status']} "
      f"observation={'None(拒绝)' if br3['observation'] is None else '存在'}")
