"""HeroCards 恢复 — 审计校验：合法 GT 计数 + validate_gt。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

GT_PATH = Path("data/vision2/coinpoker_1876_gt_herocards.json")
EV = "data/vision2/coinpoker_evidence"
gt = json.loads(GT_PATH.read_text(encoding="utf-8"))

from poker_decision_ai.vision2.pixel.realclient.gt_loader import (  # noqa: E402
    validate_gt, gt_summary)

errs = validate_gt(gt, EV)
print(f"validate_gt errors: {errs}")

valid = [a for a in gt["annotations"] if a.get("hero_card_visual", {}).get(
    "provenance_status") == "VALID"]
retr = [a for a in gt["annotations"] if a.get("hero_card_visual", {}).get(
    "provenance_status") != "VALID"]
print(f"VALID = {len(valid)}; RETRACTED = {len(retr)}")
hands = set()
for a in valid:
    cs = a["fields"]["hero_cards"]
    hands.add(tuple(cs))
    print(f"  VALID: {a['frame_id'][:32]} {cs} "
          f"model={a['hero_card_visual']['visual_model']} "
          f"conf={a['hero_card_visual']['visual_confirmation']}")
print(f"独立手牌 = {len(hands)}: {sorted(hands)}")
print(f"\n证据门（≥3 独立合法 GT）: "
      f"{'PASS' if len(hands) >= 3 else 'NOT MET'}")
