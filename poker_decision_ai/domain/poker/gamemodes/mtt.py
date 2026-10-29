"""普通 MTT 插件：共享 Agents + ICM / 阶段 / 盲注压力 / ICM 剥削。

Bubble / Final Table / Pay Jump / Heads-up 相关逻辑由 Tournament Stage Agent
按阶段动态给出上下文（不硬编码在插件里）。
"""
from __future__ import annotations

from typing import ClassVar

from poker_decision_ai.core.gamemode.plugin import GameModePlugin

from . import base as S


class MttPlugin(GameModePlugin):
    mode_id: ClassVar[str] = "tournament"
    type_id: ClassVar[str] = "mtt"
    display_name: ClassVar[str] = "普通 MTT"
    supported_modes: ClassVar[list[str]] = ["tournament:mtt"]
    required_state_fields: ClassVar[list[str]] = ["hand_id", "game_type", "street", "tournament"]

    def plan_stages(self):
        return [
            S.perception_stage(),
            S.tournament_context_stage(),
            S.icm_stage(),
            S.human_models_a(population_extra={"tournament_stage": "$tournament_stage"}),
            S.human_models_b(),
            S.range_stage(stage_extra={"tournament_stage": "$tournament_stage"}),
            S.equity_stage(),
            S.frequency_blend_stage(),
            S.theory_stage(risk_extra={"icm": "$icm", "blind_pressure": "$blind_pressure"}),
            S.exploit_bluff_stage(
                icm_exploit=True,
                bluff_extra={"icm": "$icm", "tournament_stage": "$tournament_stage"}),
            S.debate_stage(with_icm=True),
            S.judge_stage(self),
            S.deviation_stage(),
            S.post_stage(),
        ]

    def extra_judge_inputs(self) -> dict:
        return {
            "icm": "$icm",
            "tournament_stage": "$tournament_stage",
            "satellite": "$satellite",
            "icm_exploit": "$icm_exploit",
            "blind_pressure": "$blind_pressure",
        }
