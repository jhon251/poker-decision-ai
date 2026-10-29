"""P29-B — GT 校验 + 合法计数（恢复后）。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

GT_PATH = Path("data/vision2/coinpoker_1876_gt_herocards.json")
EV = "data/vision2/coinpoker_evidence"
gt = json.loads(GT_PATH.read_text(encoding="utf-8"))

from poker_decision_ai.vision2.pixel.realclient.gt_loader import (  # noqa: E402
    validate_gt)
print(f"validate_gt errors: {validate_gt(gt, EV)}")
valid = [a for a in gt["annotations"] if a.get("hero_card_visual", {}).get(
    "provenance_status") == "VALID"]
hands = set()
for a in valid:
    hands.add(tuple(a["fields"]["hero_cards"]))
    print(f"  VALID: {a['frame_id'][:30]} {a['fields']['hero_cards']}")
print(f"VALID={len(valid)} 独立手牌={len(hands)}: {sorted(hands)}")
print(f"证据门(≥3): {'PASS' if len(hands) >= 3 else 'NOT MET'}")
