"""PACKAGE+UI — runner 链 smoke：给定帧 → 一键链 → AnalysisResult。"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, '.')
from poker_decision_ai.vision2_platform.png_decode import decode_png
from cash_game_ui.runner import CashGameRunner

EV = Path("data/vision2/coinpoker_evidence")
w, h, rgba = decode_png((EV / "8bf82f134651db0add626be0304a2ff1.png").read_bytes())

r = CashGameRunner()
res = r.analyze(rgba, w, h)
d = res.to_dict()
print("status:", d["status"])
print("action:", d["decision_action"])
print("solver:", d["solver_mode"], "| strategy:", d["strategy_mode"])
print("agents:", [(a["agent"], a["status"]) for a in d["agents"]])
print("vision:", d["vision"])
print("blocking:", d["blocking_fields"], "| reason:", d["reason"])
