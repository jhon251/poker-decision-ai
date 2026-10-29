"""P30-2A — 下注状态视觉证据 GT 记录（READ-ONLY 阶段产物）。

字段:
- to_call: 视觉来源 = hero 动作面板「跟注 N」按钮金额（pc110 跟注 2,000）
- current_bet: 视觉来源 = 动作面板「加注 N」按钮金额（raise-to；8bf8 加注 4,000）
  —— 注意：当前只有"按钮显示金额"，语义=动作目标额；桌级 current_bet
  需区分"当前已下注最大额"（未观察到独立显示——per-seat bet capsule 已有）
- sb/bb: 视觉来源 = 顶左 header 文本「1K/2K」(NLH <hand_id> - 1K/2K (320))
- ante: 视觉来源 = header「(320)」
- 币种/桌信息 = header 同区
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

GT_PATH = Path("data/vision2/coinpoker_1876_gt_betting_state.json")
EV = "data/vision2/coinpoker_evidence"

ANNOT = [
    {
        "frame_id": "POT_CHANGED_110_1787992526057_full.png",
        "field": "to_call",
        "visual_bbox": [1250, 1150, 606, 240],   # 动作面板右下（目视校准）
        "amount_bbox": [1552, 1237, 100, 60],     # 「跟注 2,000」金额行
        "visual_description": "hero 动作按钮「跟注 2,000」（白色），"
                              "X/F、跟注任何按钮同列；金额=2,000",
        "ground_truth_value": 2000,
        "provenance": "image_observed (V4 Flash; 2026-08-31)",
        "visibility": "NORMAL",
        "confounders": ("与加注/出注金额、pot、stack 数值形状同（金色/白色）"
                        "——需按钮上下文分离；过牌按钮存在时 to_call=0 无金额"),
        "normal_or_dim": "NORMAL",
        "status": "candidate_gt",
    },
    {
        "frame_id": "8bf82f134651db0add626be0304a2ff1.png",
        "field": "current_bet_raise_to",
        "visual_bbox": [1250, 1150, 606, 240],
        "amount_bbox": [1414, 1288, 145, 55],     # 「加注 4,000」金额行
        "visual_description": "动作按钮「加注 4,000」+ 滑动条 4,000 + "
                              "33%/50%/75%/最大 预设；金额=4,000（加注目标额）",
        "ground_truth_value": 4000,
        "provenance": "image_observed (V4 Flash; 2026-08-31)",
        "visibility": "NORMAL",
        "confounders": ("raise-to 语义≠当前已下注最大额（current_bet）；"
                        "同区 4,000 出现于滑块与按钮两处",
                        "过牌可用 → to_call=0（本帧但仍有加注金额）"),
        "normal_or_dim": "NORMAL",
        "status": "candidate_gt",
    },
    {
        "frame_id": "8bf82f134651db0add626be0304a2ff1.png",
        "field": "sb",
        "visual_bbox": [9, 7, 176, 14],           # header 文本（全帧稳定）
        "visual_description": "顶左 header「NLH 1227160 - 1K/2K (320)」；"
                              "1K=SB（小盲）",
        "ground_truth_value": 1000,
        "provenance": "image_observed (V4 Flash; 2026-08-31)",
        "visibility": "NORMAL",
        "confounders": ("与 hand_id「1227160」同区文本；K/M 后缀需解析；"
                        "header 含手牌号干扰"),
        "normal_or_dim": "NORMAL",
        "status": "candidate_gt",
    },
    {
        "frame_id": "8bf82f134651db0add626be0304a2ff1.png",
        "field": "bb",
        "visual_bbox": [9, 7, 176, 14],
        "visual_description": "header「1K/2K (320)」；2K=BB（大盲）",
        "ground_truth_value": 2000,
        "provenance": "image_observed (V4 Flash; 2026-08-31)",
        "visibility": "NORMAL",
        "confounders": ("K 后缀金额解析；与 hand_id 相邻"),
        "normal_or_dim": "NORMAL",
        "status": "candidate_gt",
    },
    {
        "frame_id": "8bf82f134651db0add626be0304a2ff1.png",
        "field": "ante",
        "visual_bbox": [9, 7, 176, 14],
        "visual_description": "header「(320)」括号值=ante",
        "ground_truth_value": 320,
        "provenance": "image_observed (V4 Flash; 2026-08-31)",
        "visibility": "NORMAL",
        "confounders": ("需括号上下文分离（同区无括号 = 非 ante）"),
        "normal_or_dim": "NORMAL",
        "status": "candidate_gt",
    },
    {
        "frame_id": "BUTTON_CHANGED_167_1787992682749_full.png",
        "field": "sb",
        "visual_bbox": [9, 7, 176, 14],
        "visual_description": "「NLH 1218466 - 1K/2K (320)」同格式（先手手番）",
        "ground_truth_value": 1000,
        "provenance": "image_observed (V4 Flash; 2026-08-31)",
        "visibility": "NORMAL",
        "confounders": "同上",
        "normal_or_dim": "NORMAL",
        "status": "candidate_gt",
    },
    {
        "frame_id": "8bf82f134651db0add626be0304a2ff1.png",
        "field": "sb_role_badge",
        "visual_bbox": [138, 326, 52, 40],
        "visual_description": "S1 座左上蓝「SB」徽章=小盲角色（非金额）",
        "ground_truth_value": "SB",
        "provenance": "image_observed (V4 Flash; 2026-08-31)",
        "visibility": "NORMAL",
        "confounders": ("与金额无关（角色徽章），不得当金额读"),
        "normal_or_dim": "NORMAL",
        "status": "candidate_gt",
    },
    {
        "frame_id": "8bf82f134651db0add626be0304a2ff1.png",
        "field": "bb_role_badge",
        "visual_bbox": [816, 211, 51, 40],
        "visual_description": "S2 座左上蓝「BB」徽章=大盲角色（非金额）",
        "ground_truth_value": "BB",
        "provenance": "image_observed (V4 Flash; 2026-08-31)",
        "visibility": "NORMAL",
        "confounders": "与金额无关",
        "normal_or_dim": "NORMAL",
        "status": "candidate_gt",
    },
]

gt = {
    "schema_version": "v2",
    "render_class": "1876",
    "annotation_source": "P30-2A image-capable visual verification (READ-ONLY)",
    "ground_truth_status": "candidate_gt_not_production",
    "note": ("P30-2A 下注状态视觉证据：to_call/current_bet=动作面板按钮金额；"
             "sb/bb/ante=顶左 header 文本；SB/BB 蓝徽章=角色（非金额）。"
             "全部为本阶段候选 GT——未经检测器验证，不得用于生产监督。"),
    "annotations": [],
}
# validate_gt 格式：用冻结 load 语义（字段白名单），存为独立证据文件——
# 本文件是证据记录，非 22-GT；含 visual 字段扩展（不被 validate_gt 消费）。
GT_PATH.write_text(json.dumps(gt | {"evidence": ANNOT}, indent=2,
                              ensure_ascii=False), encoding="utf-8")
print(f"written -> {GT_PATH}")
print(f"evidence records = {len(ANNOT)}")
