"""KO（固定赏金 Knockout）插件：Pipeline 与 PKO 相同，赏金无递增。

赏金 Agent 内部根据 bounty_type/state.tournament.progressive 区分递增逻辑。
"""
from __future__ import annotations

from typing import ClassVar

from .bounty import BountyPkoPlugin


class BountyKoPlugin(BountyPkoPlugin):
    type_id: ClassVar[str | None] = "ko"
    display_name: ClassVar[str] = "KO（固定赏金 Knockout）"
    supported_modes: ClassVar[list[str]] = ["bounty:ko"]
    decision_modifiers: ClassVar[dict] = {
        "objective": "chip_ev_plus_bounty_ev_plus_icm",
        "use_bounty_adjusted_equity": True,
        "progressive": False,
    }
