"""P29-A — HeroCards GT 记录生成：5 样本（3 独立新 + 1 现有 + 1 overlay 变体）。
每样本：frame_id / hero_seat=S6 / card_1 / card_2 / bbox / 视觉描述 / provenance。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

GT_PATH = Path("data/vision2/coinpoker_1876_gt_herocards.json")
ev = "data/vision2/coinpoker_evidence"

# 卡片几何（P29-A 实测）：双卡并排重叠，总区 (795,1011)-(1060,1188)
# 卡1 左缘 795-813、卡2 右缘 1060；卡宽实测卡1≈121px 卡2≈130px（重叠 ~10px）
# rank 角标：左上 (795-875, 1010-1075)；suit：下方 (795-880, 1050-1130)
SAMPLES = [
    {
        "frame_id": "POT_CHANGED_102_1787992505794_full.png",
        "hero_seat": "S6",
        "card_1": "9c", "card_2": "3d",
        "card_bbox_1": [812, 1011, 121, 178],
        "card_bbox_2": [836, 1011, 123, 178],
        "pair_bbox": [812, 1011, 248, 178],
        "visual_condition": "dim-normal (global p50≈10.8) + poop-overlay 部分遮挡（rank 可读）",
        "rank_visual": "黑 9（左上角，黑墨）",
        "suit_visual": "黑梅花（左）红方块（右）",
        "provenance": "user GT (existing) + P29-A image-capable 目视复核",
        "gt_status": "existing (user-annotated, re-verified)",
    },
    {
        "frame_id": "POT_CHANGED_110_1787992526057_full.png",
        "hero_seat": "S6",
        "card_1": "6c", "card_2": "3h",
        "card_bbox_1": [813, 1011, 121, 178],
        "card_bbox_2": [837, 1011, 122, 178],
        "pair_bbox": [813, 1011, 246, 178],
        "visual_condition": "NORMAL（干净无遮；dealer D 左上可见）",
        "rank_visual": "黑 6（左上角）",
        "suit_visual": "黑梅花（左）红桃（右，大红桃）",
        "provenance": "P29-A corpus scan (white two-card cluster 813-1059) + 目视",
        "gt_status": "new",
    },
    {
        "frame_id": "POT_CHANGED_16_1787995303501_full.png",
        "hero_seat": "S6",
        "card_1": "Kc", "card_2": "3h",
        "card_bbox_1": [813, 1011, 121, 178],
        "card_bbox_2": [837, 1011, 122, 178],
        "pair_bbox": [813, 1011, 246, 178],
        "visual_condition": "NORMAL（干净无遮；blue light 上沿）",
        "rank_visual": "黑 K（左上角）",
        "suit_visual": "黑梅花（左）红桃（右）",
        "provenance": "P29-A corpus scan + 目视（不同手牌：K♣3♥ vs 6♣3♥ vs 9♣3♦）",
        "gt_status": "new",
    },
    {
        "frame_id": "POT_CHANGED_143_1787992603846_full.png",
        "hero_seat": "S6",
        "card_1": "6c", "card_2": "3h",
        "card_bbox_1": [813, 1011, 121, 178],
        "card_bbox_2": [837, 1011, 122, 178],
        "pair_bbox": [813, 1011, 246, 178],
        "visual_condition": "NORMAL（干净；等价于 pc110 同手重复样本）",
        "rank_visual": "黑 6（左上角）",
        "suit_visual": "黑梅花（左）红桃（右）",
        "provenance": "P29-A corpus scan + 目视",
        "gt_status": "new (repeat hand of pc110)",
    },
    {
        "frame_id": "SCENE_CHANGED_74_1787992438991_full.png",
        "hero_seat": "S6",
        "card_1": "9c", "card_2": "3d",
        "card_bbox_1": [813, 1011, 121, 178],
        "card_bbox_2": [837, 1011, 123, 178],
        "pair_bbox": [813, 1011, 246, 178],
        "visual_condition": "dim-normal + poop/fly overlay 强遮挡（rank 仍可读，中部被遮）",
        "rank_visual": "黑 9（左上角清晰）",
        "suit_visual": "黑梅花（左）红方块（右，被遮部分）",
        "provenance": "P29-A corpus scan (835-1077 variant) + 目视",
        "gt_status": "new (overlay variant of f03 hand)",
    },
]

gt = {
    "schema_version": "v2",
    "render_class": "1876",
    "annotation_source": "user + P29-A image-capable visual verification",
    "ground_truth_status": "user_annotated + verified",
    "note": ("HeroCards GT: 5 样本（3 独立手牌：9♣3♦ / 6♣3♥ / K♣3♥ + "
             "重复与 overlay 变体）。卡片并排重叠显示于 hero S6 卡区 "
             "(795-1060, 1011-1188)；rank 左上角、suit 下方。hero 恒 S6 "
             "（P28-B CLOSED）。"),
    "annotations": SAMPLES,
}
GT_PATH.write_text(json.dumps(gt, indent=2, ensure_ascii=False),
                   encoding="utf-8")
print(f"written -> {GT_PATH}")
print(f"samples = {len(SAMPLES)}（独立手牌 = 3）")

# 用冻结 gt_loader 校验字段合法性
from poker_decision_ai.vision2.pixel.realclient.gt_loader import (  # noqa: E402
    validate_gt, gt_summary)
errs = validate_gt(gt, ev)
print(f"validate_gt errors: {errs}")
