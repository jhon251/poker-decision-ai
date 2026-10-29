"""P29-A — HeroCards GT 记录 v2（冻结 schema：geometry + fields.hero_cards）。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

GT_PATH = Path("data/vision2/coinpoker_1876_gt_herocards.json")
ev = "data/vision2/coinpoker_evidence"

SAMPLES = [
    ("POT_CHANGED_102_1787992505794_full.png", ["9c", "3d"],
     "dim-normal + poop-overlay 部分遮挡（rank 可读）；黑9/黑梅花 + 红3/红方块",
     "existing (user-annotated, re-verified)",
     (812, 1011, 121, 178), (836, 1011, 123, 178), (812, 1011, 248, 178)),
    ("POT_CHANGED_110_1787992526057_full.png", ["6c", "3h"],
     "NORMAL 干净；黑6/黑梅花 + 红3/红桃",
     "new (P29-A corpus scan + 目视)",
     (813, 1011, 121, 178), (837, 1011, 122, 178), (813, 1011, 246, 178)),
    ("POT_CHANGED_16_1787995303501_full.png", ["Kc", "3h"],
     "NORMAL 干净；黑K/黑梅花 + 红3/红桃（独立手牌 K♣3♥）",
     "new (corpus scan + 目视)",
     (813, 1011, 121, 178), (837, 1011, 122, 178), (813, 1011, 246, 178)),
    ("POT_CHANGED_143_1787992603846_full.png", ["6c", "3h"],
     "NORMAL 干净；同 pc110 手牌（6♣3♥ 重复样本）",
     "new (repeat hand of pc110)",
     (813, 1011, 121, 178), (837, 1011, 122, 178), (813, 1011, 246, 178)),
    ("SCENE_CHANGED_74_1787992438991_full.png", ["9c", "3d"],
     "dim-normal + poop/fly overlay 强遮挡（rank 仍可读）；同 f03 手牌",
     "new (overlay variant of f03 hand)",
     (813, 1011, 121, 178), (837, 1011, 123, 178), (813, 1011, 246, 178)),
]

anns = []
for fid, cards, desc, prov, b1, b2, pb in SAMPLES:
    anns.append({
        "frame_id": fid,
        "geometry": [1876, 1400],
        "fields": {
            "hero_cards": cards,
            "position": {"dealer": {"seat": "S1", "button_bbox": [1, 1, 1, 1]},
                         "occupied": {f"S{i}": "occupied" for i in range(1, 7)},
                         "hero": "S6"},
            "stack": None, "status": None, "pot": None, "bet": None,
        },
        "hero_card_visual": {
            "hero_seat": "S6",
            "card_bbox_1": list(b1),
            "card_bbox_2": list(b2),
            "pair_bbox": list(pb),
            "visual_description": desc,
            "rank_zone": [795, 1010, 80, 65],
            "suit_zone": [795, 1050, 85, 80],
            "provenance": prov,
        },
    })

gt = {
    "schema_version": "v2",
    "render_class": "1876",
    "annotation_source": "user + P29-A image-capable visual verification",
    "ground_truth_status": "user_annotated + verified",
    "note": ("HeroCards GT: 5 样本 / 3 独立手牌（9♣3♦ / 6♣3♥ / K♣3♥）。"
             "卡片并排重叠于 hero S6 卡区 (≈795-1060, 1011-1188)；rank 左上角"
             "(795,1010,80,65)、suit 下方 (795,1050,85,80)。rank/suit 视觉可区分"
             "（NORMAL 清晰；overlay 帧 rank 可读）。hero 恒 S6（P28-B CLOSED）。"),
    "annotations": anns,
}
GT_PATH.write_text(json.dumps(gt, indent=2, ensure_ascii=False),
                   encoding="utf-8")
print(f"written -> {GT_PATH}")

from poker_decision_ai.vision2.pixel.realclient.gt_loader import (  # noqa: E402
    validate_gt, gt_summary)
errs = validate_gt(gt, ev)
print(f"validate_gt errors: {errs}")
print(f"gt_summary hero_cards: "
      f"{gt_summary(gt)['hero_cards']}")
