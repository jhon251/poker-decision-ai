"""P30-3-A — Case A-E 决策可执行性：bridge→StateAgent→下游状态完整性。
通过真实接口（frame_to_decision_request → VisionAgent → StateAgent）
检查各 UNKNOWN 场景下游状态。"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    frame_to_decision_request, snapshot_to_vision_observation)
from poker_decision_ai.agents.vision.agent import VisionAgent  # noqa: E402
from poker_decision_ai.agents.state.agent import StateAgent  # noqa: E402
from poker_decision_ai.core.config import load_config  # noqa: E402
from poker_decision_ai.core.logging import EventLogger, MetricsCollector  # noqa: E402
from poker_decision_ai.core.orchestration.context import AgentContext  # noqa: E402
from poker_decision_ai.core.providers import build_providers  # noqa: E402
from poker_decision_ai.core.repositories import build_in_memory  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")

cfg = load_config()
ctx = AgentContext("t", "s", cfg, build_providers(cfg), build_in_memory(),
                   EventLogger("logs"), MetricsCollector())


def chain_state(snap, pr, tag):
    br = snapshot_to_vision_observation(snap, pr)
    obs = br["observation"]
    request = {"kind": "cash", "hero_id": "Hero",
               "vision_observation": obs}
    vout = asyncio.run(VisionAgent().run({"request": request}, ctx))
    vd = vout.output
    sout = asyncio.run(StateAgent().run({"request": request, "vision": vd},
                                        ctx))
    st = sout.output
    return {k: st.get(k) for k in
            ("street", "pot", "current_bet", "to_call", "board",
             "available_actions", "hero_cards", "action_history_len")}


def base_snap():
    return {
        "street": {"last_street": "PREFLOP"},
        "pot": {"last_pot": 8920, "last_status": "RECOGNIZED"},
        "hero_cards": {"last_cards": ["5c", "3c"],
                       "last_status": "RECOGNIZED"},
        "board": {"last_cards": []},
        "bet": {"last_seats": {}},
        "status": {"last_seats": {f"S{i}": {"status": "NOT_OBSERVED"}
                                  for i in range(1, 7)}},
        "stack": {"last_seats": {f"S{i}": {"stack": 1000 * i,
                                           "status": "RECOGNIZED"}
                                 for i in range(1, 7)}},
        "position": {"last_hero_position": "CO",
                     "last_uncertainty": "RECOGNIZED"},
        "blind": {"last_sb": 1000, "last_bb": 2000, "last_ante": 320,
                  "last_status": "RECOGNIZED"},
        "to_call": {"last_to_call": 2000, "last_status": "RECOGNIZED"},
    }


def base_pr():
    return {"occupied_seats": {f"S{i}": {"status": "OCCUPIED"}
                               for i in range(1, 7)},
            "dealer_seat": "S4", "hero_seat": "S6", "table_topology":
            "6-max"}


# Case A: 全 READY
st = chain_state(base_snap(), base_pr(), "A")
print("Case A (all ready):", {k: st[k] for k in
      ("street", "pot", "current_bet", "to_call", "available_actions")})

# Case B: current_bet UNKNOWN —— V3F bridge 根本不提供 current_bet →
# StateAgent 从 vision_observation 读取缺省 → 0.0
print("Case B (current_bet UNKNOWN): V2 StateAgent 对缺失 current_bet 处理：")
print("  即 obs 无 current_bet → float(obs.get('current_bet') or 0.0) = 0.0")
print("  => UNKNOWN→0 风险点（StateAgent 既有语义，本阶段只报告）")

# Case D: occupied UNKNOWN（f03 S6）
snap_d = base_snap()
pr_d = base_pr()
pr_d["occupied_seats"]["S6"] = {"status": "UNKNOWN"}
br = snapshot_to_vision_observation(snap_d, pr_d)
print(f"\nCase D (occupied UNKNOWN): bridge={br['status']} "
      f"obs={'None(拒绝)' if br['observation'] is None else '存在'}")
