"""P30-2C-A — to_call GT 证据记录（3 独立状态：2,000 / 124,163 / 0-check）。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

GT_PATH = Path("data/vision2/coinpoker_1876_gt_to_call.json")

ANNOT = [
    {
        "frame_id": "POT_CHANGED_110_1787992526057_full.png",
        "field": "to_call",
        "visual_object": "hero action panel「跟注 N」按钮",
        "button_bbox": [1350, 1230, 320, 170],     # 炭灰按钮（估）
        "amount_text": "2,000",
        "ground_truth_value": 2000,
        "provenance": "image_observed (V4 Flash; 2026-08-31)",
        "normal_or_dim": "NORMAL", "visibility": "full",
        "confounders": ("「加注 4,000」同区（raise-to ≠ to_call）；"
                        "「跟注任何」无金额；「X/F」=check/fold 组合"),
        "status": "candidate_gt",
    },
    {
        "frame_id": "POT_CHANGED_114_1787992536082_full.png",
        "field": "to_call",
        "visual_object": "hero action panel「跟注 N」按钮（绿色）",
        "button_bbox": [1610, 1265, 450, 175],      # 绿按钮（目视校准）
        "amount_text": "124,163",
        "ground_truth_value": 124163,
        "provenance": "image_observed (V4 Flash; 2026-08-31)",
        "normal_or_dim": "NORMAL", "visibility": "full",
        "confounders": ("红「盖牌」钮左邻（fold）；绿色按钮样式与"
                        "「过牌」同为绿——需金额行存在性区分"),
        "status": "candidate_gt",
    },
    {
        "frame_id": "POT_CHANGED_177_1787992718375_full.png",
        "field": "to_call",
        "visual_object": "hero action panel「过牌」（check 按钮）",
        "button_bbox": [1440, 1230, 300, 170],
        "amount_text": None,
        "ground_truth_value": 0,
        "provenance": "image_observed (V4 Flash; 2026-08-31)",
        "normal_or_dim": "NORMAL", "visibility": "full",
        "note": ("过牌=无需跟注（视觉语义直接表达，非扑克规则推导）："
                 "check 按钮无金额行"),
        "confounders": ("「跟注任何」+「X/F」同区；「盲抓」绿色按钮"
                        "（pc14——非 过牌，无金额行但语义不同 → 需区分）"),
        "status": "candidate_gt",
    },
]

gt = {
    "schema_version": "v2",
    "render_class": "1876",
    "annotation_source": "P30-2C-A image-observed visual verification",
    "ground_truth_status": "candidate_gt_not_production",
    "note": ("to_call 视觉证据：跟注按钮金额（2,000 / 124,163）+"
             "过牌=0（直接视觉语义）。3 独立状态——证据门满足。"
             "加注金额=raise-to ≠ to_call（禁止混淆）。"),
    "annotations": [],
    "evidence": ANNOT,
}
GT_PATH.write_text(json.dumps(gt | {"evidence": ANNOT}, indent=2,
                              ensure_ascii=False), encoding="utf-8")
print(f"written -> {GT_PATH} ({len(ANNOT)} evidence records)")
