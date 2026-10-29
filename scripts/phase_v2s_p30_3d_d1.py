"""P30-3-D D1 — 真实 DecisionInput 全链审计：8bf8 frame → StateAgent 输出
→ Judge 实际输入（equity/solver/exploit 源 + state 字段）。"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    snapshot_to_vision_observation)
from poker_decision_ai.vision2.pixel.realclient.runtime_snapshot import (  # noqa: E402
    build_snapshot)
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    PositionObserver, PotObserver, BoardObserver, StreetObserver,
    HeroCardsObserver, StatusObserver, StackObserver)
from poker_decision_ai.vision2.pixel.realclient.bet_observer import (  # noqa: E402
    BetObserver)
from poker_decision_ai.vision2.pixel.realclient.betting_header_observer import (  # noqa: E402
    BettingHeaderObserver)
from poker_decision_ai.vision2.pixel.realclient.to_call_observer import (  # noqa: E402
    ToCallObserver)
from poker_decision_ai.agents.vision.agent import VisionAgent  # noqa: E402
from poker_decision_ai.agents.state.agent import StateAgent  # noqa: E402
from poker_decision_ai.core.config import load_config  # noqa: E402
from poker_decision_ai.core.orchestration.context import AgentContext  # noqa: E402
from poker_decision_ai.core.providers import build_providers  # noqa: E402
from poker_decision_ai.core.repositories import build_in_memory  # noqa: E402
from poker_decision_ai.core.logging import EventLogger, MetricsCollector  # noqa: E402
import inspect

EV = Path("data/vision2/coinpoker_evidence")
cfg = load_config()
ctx = AgentContext("t", "s", cfg, build_providers(cfg), build_in_memory(),
                   EventLogger("logs"), MetricsCollector())

w, h, rgba = decode_png((EV /
    "8bf82f134651db0add626be0304a2ff1.png").read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)

obs = {
    "position_runtime": PositionObserver(), "pot_runtime": PotObserver(),
    "board_runtime": BoardObserver(), "street_runtime": StreetObserver(),
    "hero_cards_runtime": HeroCardsObserver(),
    "status_runtime": StatusObserver(), "stack_runtime": StackObserver(),
    "bet_runtime": BetObserver(),
    "betting_header_runtime": BettingHeaderObserver(),
    "to_call_runtime": ToCallObserver(),
}
for o in obs.values():
    o.observe(arr, w, h)
snap = build_snapshot(obs)
snap["blind"] = obs["betting_header_runtime"].summary()
snap["to_call"] = obs["to_call_runtime"].summary()
br = snapshot_to_vision_observation(
    snap, (obs["position_runtime"].last or {}).get("runtime_result") or {})
print(f"bridge: {br['status']}")
ob = br["observation"]
request = {"kind": "cash", "hero_id": "Hero", "vision_observation": ob}
vout = asyncio.run(VisionAgent().run({"request": request}, ctx))
sout = asyncio.run(StateAgent().run({"request": request, "vision": vout},
                                    ctx))
st = sout.output
print("\n=== StateAgent 输出（PokerState→dict）===")
for k in ("street", "pot", "current_bet", "to_call", "hero_seat",
          "button_seat", "sb", "bb", "ante", "min_raise_to",
          "available_actions", "last_aggressor"):
    print(f"  {k} = {getattr(st, k, '?')}")
players = getattr(st, "players", [])
print(f"  players ({len(players)}):")
for p in players:
    print(f"    {p['seat']}: {p['name']} stack={p['stack']} "
          f"status={p['status']} hero={p['is_hero']}")
print(f"  hero_cards = {getattr(st, 'hero_cards', '?')}")
print(f"  board = {getattr(st, 'board', '?')}")
print(f"  action_history len = {len(getattr(st, 'action_history', []))}")
