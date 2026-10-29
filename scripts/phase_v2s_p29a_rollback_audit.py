"""P29-A integrity rollback — 审计校验脚本：
1) validate_gt 通过（含 4 个 INVALID_RETRACTED 条目的结构化字段）
2) 合法 hero_cards 计数 = 1（仅 POT_CHANGED_102）
3) 确认无 HeroCards production code 产生
4) 报告回滚前后状态
"""
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

# 合法计数（provenance_status == VALID）
valid = [a for a in gt["annotations"]
         if a.get("hero_card_visual", {}).get("provenance_status") == "VALID"]
retracted = [a for a in gt["annotations"]
             if a.get("hero_card_visual", {}).get("provenance_status")
             == "INVALID_RETRACTED"]
print(f"VALID annotations = {len(valid)}")
print(f"RETRACTED annotations = {len(retracted)}")
for a in valid:
    print(f"  VALID: {a['frame_id']} -> {a['fields']['hero_cards']} "
          f"({a['hero_card_visual']['provenance']})")
for a in retracted:
    print(f"  RETRACTED: {a['frame_id']} -> "
          f"{a['fields']['hero_cards']} "
          f"({a['hero_card_visual']['retraction_reason'][:40]}...)")

# validate 仅看 fields.hero_cards——retracted 也含 hero_cards 值但
# provenance_status 标记无效；gt_summary 计数展示
s = gt_summary(gt)
print(f"\ngt_summary hero_cards: {s['hero_cards']}")
print(f"（注：gt_summary 按字段计 annotated，不区分 valid/retracted——"
      f"合法身份计数以 provenance_status=VALID 为准 = {len(valid)}）")

# 无 production code 确认
prod_patterns = [
    ("herocards_detector.py", Path("poker_decision_ai/vision2/pixel/realclient/"
                                   "herocards_detector.py").exists()),
    ("hero_cards 1876 recognizer grep", "HeroCardsRecognizer1876"),
]
for name, flag in prod_patterns:
    print(f"production code check [{name}]: {'EXISTS' if flag else 'none'}")

# 无 1876 hero cards 分支引入（seat_position.py 中 hero_cards 1876 分支应为 EVIDENCE_LIMITED）
src = Path("poker_decision_ai/vision2/pixel/realclient/seat_position.py").read_text(
    encoding="utf-8")
print("seat_position.py hero_cards_1876_layout_not_extracted still present:",
      "hero_cards_1876_layout_not_extracted" in src)
