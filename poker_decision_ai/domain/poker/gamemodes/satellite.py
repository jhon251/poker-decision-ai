"""卫星赛插件：共享 Agents + ICM + Satellite（最高优先级）+ 阶段 + 盲注压力。

目标：最大化席位/晋级概率（Seat EV），而不是最大化筹码 EV。
不启用 ICM Exploit（普通 MTT 的 bubble 逻辑不直接套用在卫星赛）。
"""
from __future__ import annotations

from typing import ClassVar

from poker_decision_ai.core.gamemode.plugin import GameModePlugin

from . import base as S


class SatellitePlugin(GameModePlugin):
    mode_id: ClassVar[str] = "tournament"
    type_id: ClassVar[str] = "satellite"
    display_name: ClassVar[str] = "卫星赛"
    supported_modes: ClassVar[list[str]] = ["tournament:satellite"]
    required_state_fields: ClassVar[list[str]] = ["hand_id", "game_type", "street", "tournament"]
    #: Judge 修饰器：卫星赛客观目标 = 席位概率，ICM 常规逻辑只作参考
    decision_modifiers: ClassVar[dict] = {
        "satellite_priority": True,
        "objective": "seat_probability",
    }
    evaluation_rules: ClassVar[dict] = {"objective": "seat_probability"}

    def plan_stages(self):
        return [
            S.perception_stage(),
            S.tournament_context_stage(),
            S.icm_stage(),
            S.satellite_stage(),          # 紧跟 ICM，优先级最高
            S.human_models_a(population_extra={"tournament_stage": "$tournament_stage"}),
            S.human_models_b(),
            S.range_stage(stage_extra={"tournament_stage": "$tournament_stage"}),
            S.equity_stage(),
            S.theory_stage(risk_extra={"icm": "$icm", "blind_pressure": "$blind_pressure"}),
            S.exploit_bluff_stage(
                icm_exploit=False,
                bluff_extra={"icm": "$icm", "tournament_stage": "$tournament_stage"}),
            S.debate_stage(with_icm=True),
            S.judge_stage(self),
            S.deviation_stage(),
            S.post_stage(),
        ]

    def extra_judge_inputs(self) -> dict:
        return {
            "icm": "$icm",
            "satellite": "$satellite",
            "tournament_stage": "$tournament_stage",
            "blind_pressure": "$blind_pressure",
        }
