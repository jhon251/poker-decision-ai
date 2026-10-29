"""Mystery Bounty 插件：PKO Pipeline + Mystery Bounty Agent（预留接口）。

Mystery Bounty Agent 的 Interface/Schema/Provider/注册/测试接口 V1.1 全部建立；
其逻辑实现程度以 agent 文档为准（揭晓阶段前按固定期望值处理）。
"""
from __future__ import annotations

from typing import ClassVar

from poker_decision_ai.core.gamemode.plugin import GameModePlugin

from . import base as S
from .bounty import BountyPkoPlugin


class MysteryBountyPlugin(BountyPkoPlugin):
    type_id: ClassVar[str | None] = "mystery"
    display_name: ClassVar[str] = "Mystery Bounty"
    supported_modes: ClassVar[list[str]] = ["bounty:mystery"]
    decision_modifiers: ClassVar[dict] = {
        "objective": "chip_ev_plus_bounty_ev_plus_icm",
        "use_bounty_adjusted_equity": True,
        "mystery_phase_aware": True,
    }

    def plan_stages(self):
        stages = super().plan_stages()
        # 在 bounty_equity 之后插入 mystery_bounty（揭晓阶段期望值评估）
        out = []
        for stage in stages:
            out.append(stage)
            if stage.name == "bounty_equity":
                out.append(S.mystery_bounty_stage())
        return out

    def extra_judge_inputs(self) -> dict:
        base = super().extra_judge_inputs()
        base["mystery_bounty"] = "$mystery_bounty"
        return base
