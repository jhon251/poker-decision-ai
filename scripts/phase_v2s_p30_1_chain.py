"""P30-1 — 生产链集成验证：bridge OK observation → VisionAgent → StateAgent
（既有生产入口直通路径，不改 agent）。"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    snapshot_to_vision_observation)
from poker_decision_ai.agents.vision.agent import VisionAgent  # noqa: E402
from poker_decision_ai.agents.state.agent import StateAgent  # noqa: E402
from poker_decision_ai.core.config import load_config  # noqa: E402
from poker_decision_ai.core.logging import EventLogger, MetricsCollector  # noqa: E402
from poker_decision_ai.core.orchestration.context import AgentContext  # noqa: E402
from poker_decision_ai.core.providers import build_providers  # noqa: E402
from poker_decision_ai.core.repositories import build_in_memory  # noqa: E402

# 全已知合成 snapshot（8bf8 场景：dealer S4→6, hero S6→5, 6 座）
snap = {
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
pr = {"occupied_seats": {f"S{i}": {"status": "OCCUPIED"}
                         for i in range(1, 7)},
      "dealer_seat": "S4", "hero_seat": "S6", "table_topology": "6-max"}
br = snapshot_to_vision_observation(snap, pr)
assert br["status"] == "OK", br
obs = br["observation"]

cfg = load_config()
ctx = AgentContext("t", "s", cfg, build_providers(cfg), build_in_memory(),
                   EventLogger("logs"), MetricsCollector())
request = {"kind": "cash", "hero_id": "Hero",
           "vision_observation": obs}
vout = asyncio.run(VisionAgent().run({"request": request}, ctx))
voutd = vout.output if not isinstance(vout.output, dict) else vout.output
print("VisionAgent output:", {k: voutd[k] for k in
      ("provider", "confidence") if k in voutd})

sout = asyncio.run(StateAgent().run(
    {"request": request, "vision": voutd}, ctx))
st = sout.output
if not isinstance(st, dict):
    st = st.to_dict() if hasattr(st, "to_dict") else vars(st)
print(f"\nStateOutput: street={st.get('street')} pot={st.get('pot')} "
      f"button={st.get('button_seat')} hero_seat={st.get('hero_seat')}")
hero = next((p for p in (st.get("players") or []) if p.get("is_hero")), None)
print(f"hero player: {hero}")
assert st["hero_seat"] == 5
assert st["button_seat"] == 6
assert st["pot"] == 8920.0
assert hero and hero["position"] == "CO", hero
print("\n== 生产链验证 PASS：bridge → VisionAgent → StateAgent → CO ==")
