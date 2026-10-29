"""P30-9 — loop dedup 调试。"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, '.')
import numpy as np
from poker_decision_ai.vision2_platform.png_decode import decode_png
from poker_decision_ai.core.config import load_config
from poker_decision_ai.core.orchestration.runtime import build_runtime
from poker_decision_ai.production_decision_loop import ProductionDecisionLoop

EV = Path("data/vision2/coinpoker_evidence")
w, h, rgba = decode_png((EV / "8bf82f134651db0add626be0304a2ff1.png").read_bytes())
runtime = build_runtime(load_config())
loop = ProductionDecisionLoop(runtime, throttle_s=0.0)
r1 = loop.tick(100, rgba, w, h)
print("r1:", r1["decision_status"], r1["decision_id"], "|", r1.get("warning"))
r2 = loop.tick(100, rgba, w, h)
print("r2:", r2["decision_status"], r2["decision_id"], "|", r2.get("warning"))
print("same id:", r1["decision_id"] == r2["decision_id"])
