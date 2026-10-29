"""Mystery Bounty Prize Provider 预留接口（V1.1）。

Interface + 确定性实现（DeterministicMysteryProvider）+ 真实适配器占位。
未来接入真实信封数据源（平台 API / 手工录入）时实现
MysteryBountyPrizeProvider 并注册，Mystery Bounty Agent 零改动。
"""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field

from .base import Provider


@dataclass
class MysteryPrizeInfo:
    """神秘赏金揭晓信息。"""

    phase: str = "pending"                # pending（未揭晓）| active（揭晓阶段）
    total_pool: float = 0.0               # 剩余赏金池（$）
    envelopes_remaining: int = 0          # 剩余信封数
    expected_value: float = 0.0           # 期望信封价值（$）
    known_prizes: list = field(default_factory=list)   # 已揭晓奖品 [{player, value}]
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "phase": self.phase, "total_pool": self.total_pool,
            "envelopes_remaining": self.envelopes_remaining,
            "expected_value": self.expected_value,
            "known_prizes": list(self.known_prizes), "notes": list(self.notes),
        }


class MysteryBountyPrizeProvider(Provider):
    """神秘赏金奖品信息接口（V1.1 预留）。"""

    name = "mystery_bounty"

    @abstractmethod
    async def prize_info(self, state: dict) -> MysteryPrizeInfo:
        """根据 state 返回神秘赏金揭晓信息。"""


class DeterministicMysteryProvider(MysteryBountyPrizeProvider):
    """确定性实现：揭晓前期望值 = 剩余赏金池 / 剩余信封数。

    信封数无法从 state 得到时用剩余玩家数近似（每玩家一个信封，注释说明）。
    """

    name = "mystery_bounty_deterministic"
    is_mock = True

    async def prize_info(self, state: dict) -> MysteryPrizeInfo:
        t = state.get("tournament") if isinstance(state.get("tournament"), dict) else {}
        pool = float(t.get("total_bounty_pool") or 0.0)
        phase = "active" if t.get("mystery_phase_active") else "pending"
        envelopes = int(t.get("mystery_envelopes_remaining") or 0)
        players_remaining = int(t.get("players_remaining") or 0)
        if envelopes <= 0 and players_remaining > 0:
            envelopes = players_remaining      # 近似：每玩家一个信封
            if phase == "active":
                return MysteryPrizeInfo(
                    phase=phase, total_pool=pool, envelopes_remaining=envelopes,
                    expected_value=pool / envelopes if envelopes else 0.0,
                    notes=["信封数按剩余玩家数近似（state 未提供 envelopes_remaining）"],
                )
        return MysteryPrizeInfo(
            phase=phase, total_pool=pool, envelopes_remaining=envelopes,
            expected_value=pool / envelopes if envelopes else 0.0,
        )
