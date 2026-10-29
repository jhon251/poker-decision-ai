"""P29-B 前置审计：确认合法 GT 计数 + validate_gt + 无生产代码。"""
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
retr = [a for a in gt["annotations"] if a.get("hero_card_visual", {}).get(
    "provenance_status") == "INVALID_RETRACTED"]
print(f"VALID={len(valid)} RETRACTED={len(retr)}")
for a in valid:
    print(f"  VALID: {a['frame_id'][:30]} {a['fields']['hero_cards']}")
print(f"合法 GT = {len(valid)}（证据门 ≥3 未达）")

# 生产代码审计：确认无 HeroCardsRecognizer1876 / 1876 herocards detector
import re
pkg = Path("poker_decision_ai/vision2/pixel/realclient")
hits = []
for p in pkg.glob("*.py"):
    src = p.read_text(encoding="utf-8")
    if "HeroCardsRecognizer1876" in src:
        hits.append(p.name)
print(f"HeroCardsRecognizer1876 出现文件: {hits}")
