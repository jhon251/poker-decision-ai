"""P30-8 — StateAgent field_semantics 验证（8bf8 → state）。"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    frame_to_decision_request)
from poker_decision_ai.agents.vision.agent import VisionAgent  # noqa: E402
from poker_decision_ai.agents.state.agent import StateAgent  # noqa: E402
from poker_decision_ai.core.config import load_config  # noqa: E402
from poker_decision_ai.core.orchestration.context import AgentContext  # noqa: E402
from poker_decision_ai.core.providers import build_providers  # noqa: E402
from poker_decision_ai.core.repositories import build_in_memory  # noqa: E402
from poker_decision_ai.core.logging import EventLogger, MetricsCollector  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
cfg = load_config()
ctx = AgentContext("t", "s", cfg, build_providers(cfg), build_in_memory(),
                   EventLogger("logs"), MetricsCollector())

w, h, rgba = decode_png((EV /
    "8bf82f134651db0add626be0304a2ff1.png").read_bytes())
res = frame_to_decision_request(rgba, w, h)
obs = res["bridge"]["observation"]
request = {"kind": "cash", "hero_id": "Hero", "vision_observation": obs}
vout = asyncio.run(VisionAgent().run({"request": request}, ctx))
sout = asyncio.run(StateAgent().run({"request": request, "vision": vout},
                                    ctx))
out = sout.output
print("field_semantics:", out.get("field_semantics"))
print("current_bet numeric:", out.get("current_bet"))
print("action_history:", len(out.get("action_history") or []))
