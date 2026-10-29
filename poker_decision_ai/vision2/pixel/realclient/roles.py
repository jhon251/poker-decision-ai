"""V2D — 角色级语义标注（由 ROI provenance 确定性推导；无需视觉能力）。

角色级 ≠ 字符级：
  - role（如 pot_digit_role）由 (kind, source_roi) 确定性推导——"这些资产
    是 pot 区的数字字形"，这是结构事实，不是猜；
  - semantic_label（字符身份：哪个 4/哪个 A）仍保持 null——必须由视觉
    标注者（人或 vision-capable 模型）或 OCR 完成；
  - role_identified 状态**不开启**任何值识别（后端 is_labeled 语义未变）。

P0/P1 标签定义（优先级映射到角色）：
  P0: hero_card_slot/glyph_role, board_card_slot/glyph_role,
      pot_digit_role, hero_stack(起始于 stacks_digit_role),
      bet_digit_role, button_disc_role, street(由 board 牌数推导，无资产)
  P1: stack_digit_role(=其他玩家 stacks), folded/all-in 标记(待视觉确认),
      seat/player 状态, available actions(待视觉确认), meta_digit_role
  P2: 复杂 UI 元素（current 未提取）
"""
from __future__ import annotations

from .assets import RealClientTemplateLibrary

#: (kind, source_roi) → 角色
ROLE_TABLE = {
    ("digit_glyph", "pot"): "pot_digit_role",
    ("digit_glyph", "bet_area"): "bet_digit_role",
    ("digit_glyph", "stacks"): "stack_digit_role",
    ("digit_glyph", "table_meta"): "table_meta_digit_role",
    ("card_slot", "hero_cards"): "hero_card_slot_role",
    ("card_slot", "board"): "board_card_slot_role",
    ("card_glyph", "hero_cards"): "hero_card_glyph_role",
    ("card_glyph", "board"): "board_card_glyph_role",
    ("region_snapshot", "hero_cards"): "hero_cards_region_role",
    ("region_snapshot", "board"): "board_region_role",
    ("disc_candidate", "table_meta"): "button_disc_role",
    ("disc_candidate", "players"): "player_area_disc_role",
    ("disc_candidate", "action_area"): "action_area_disc_role",
}


def role_for(kind: str, source_roi: str | None) -> str | None:
    if source_roi is None:
        return None
    return ROLE_TABLE.get((kind, source_roi))


def assign_role_labels(library: RealClientTemplateLibrary) -> int:
    """为已知 (kind, roi) 组合的资产写入 role + role_identified 状态。

    仅结构性推导；不改动 semantic_label；返回已赋角色资产数。
    """
    n = 0
    for asset in library.assets.values():
        role = role_for(asset.kind, asset.source_roi)
        if role is not None:
            asset.role = role
            if asset.ground_truth_status == "unlabeled_candidate":
                asset.ground_truth_status = "role_identified"
                asset.label_source = "roi_provenance"
            n += 1
    return n
