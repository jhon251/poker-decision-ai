"""PKO（Progressive Knockout）插件：赏金模式完整 Pipeline。

共享 Agents + ICM/阶段/盲注压力 + 四个赏金专属 Agent：
  bounty         -> 当前/对手/hero 赏金、淘汰价值、递增、换算、赏金压力
  bounty_equity  -> normal_required_equity vs bounty_adjusted_required_equity
  bounty_icm     -> ICM + Bounty EV + Risk Premium + 淘汰概率（喂 Judge）
  bounty_strategy-> 是否追大赏金/避免大赏金/对手过度追赏金/Bubble+赏金/FT+赏金
"""
from __future__ import annotations

from typing import ClassVar

from poker_decision_ai.core.gamemode.plugin import GameModePlugin

from . import base as S


class BountyPkoPlugin(GameModePlugin):
    mode_id: ClassVar[str] = "bounty"
    type_id: ClassVar[str] = "pko"
    display_name: ClassVar[str] = "PKO（Progressive Knockout）"
    supported_modes: ClassVar[list[str]] = ["bounty:pko"]
    required_state_fields: ClassVar[list[str]] = ["hand_id", "game_type", "street", "tournament"]
    decision_modifiers: ClassVar[dict] = {
        "objective": "chip_ev_plus_bounty_ev_plus_icm",
        "use_bounty_adjusted_equity": True,
    }
    evaluation_rules: ClassVar[dict] = {
        "objective": "chip_ev_plus_bounty_ev_plus_icm",
        "bounty_adjusted": True,
    }

    def plan_stages(self):
        return [
            S.perception_stage(),
            S.tournament_context_stage(),
            S.icm_stage(),
            S.bounty_stage(),
            S.human_models_a(population_extra={"tournament_stage": "$tournament_stage"}),
            S.human_models_b(),
            S.range_stage(stage_extra={"tournament_stage": "$tournament_stage"}),
            S.equity_stage(),
            S.frequency_blend_stage(),
            S.bounty_equity_stage(),
            S.bounty_icm_stage(),
            S.bounty_strategy_stage(),
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
            "tournament_stage": "$tournament_stage",
            "blind_pressure": "$blind_pressure",
            "bounty": "$bounty",
            "bounty_equity": "$bounty_equity",
            "bounty_icm": "$bounty_icm",
            "bounty_strategy": "$bounty_strategy",
        }
