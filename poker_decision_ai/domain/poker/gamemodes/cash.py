"""Cash Game 插件：共享 Poker Agents，无 ICM / 卫星 / 赏金。"""
from __future__ import annotations

from typing import ClassVar

from poker_decision_ai.core.gamemode.plugin import GameModePlugin

from . import base as S


class CashGamePlugin(GameModePlugin):
    mode_id: ClassVar[str] = "cash"
    type_id: ClassVar[str | None] = None
    display_name: ClassVar[str] = "现金桌"
    supported_modes: ClassVar[list[str]] = ["cash"]
    required_state_fields: ClassVar[list[str]] = ["hand_id", "game_type", "street"]
    optional_state_fields: ClassVar[list[str]] = ["board", "players", "pot", "to_call"]

    def plan_stages(self):
        return [
            S.perception_stage(),
            S.cash_context_stage(),
            S.human_models_a(),
            S.human_models_b(),
            S.range_stage(),
            S.equity_stage(),
            S.frequency_blend_stage(),
            S.theory_stage(),
            S.exploit_bluff_stage(icm_exploit=False),
            S.debate_stage(with_icm=False),
            S.judge_stage(self),
            S.deviation_stage(),
            S.post_stage(),
        ]
