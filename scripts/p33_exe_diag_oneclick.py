"""诊断 test_one_click_chain_ok 的 decision_action=None 根因（只读——不改任何状态）。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png

EV = Path("data/vision2/coinpoker_evidence")
w, h, rgba = decode_png((EV / "8bf82f134651db0add626be0304a2ff1.png").read_bytes())

import os as _os
_os.environ.setdefault("DSH_EVIDENCE_SOURCE", "frames_replay")  # P30-9-HQ ✓
from cash_game_ui.runner import CashGameRunner

res = CashGameRunner().analyze(rgba, w, h)
d = res.to_dict()
raw = d.get("raw") or {}
print("status         :", d.get("status"))
print("decision_action:", d.get("decision_action"))
print("raw.decision_status:", raw.get("decision_status"))
print("raw.warning        :", raw.get("warning"))
print("raw.reason         :", raw.get("reason"))
print("raw.errors         :", raw.get("errors"))
print("solver_mode    :", d.get("solver_mode"), " strategy_mode:", d.get("strategy_mode"))
judge = [a for a in d.get("agents", []) if a["agent"] == "Judge"]
if judge:
    print("judge          :", judge[0]["status"], judge[0]["output"])
# provenance 里的 gate 信息
prov = raw.get("provenance") or {}
print("prov unknown_allowed:", prov.get("unknown_allowed_fields"))
print("prov action_validated:", prov.get("action_validated"))
