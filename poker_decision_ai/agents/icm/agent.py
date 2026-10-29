"""ICM Agent: tournament dollar-equity for every remaining player.

Deterministic. Uses domain icm() / bubble_factor() / risk_premium().

IMPORTANT ASSUMPTION (documented): for ICM the economically relevant field is
every player still ALIVE in the tournament (status != "out").  A player who
folded this hand still holds chips and still competes for the remaining prize
pool, so they must stay in `stacks`.  Using only the two players active in the
current hand would collapse the field to a linear (BF=1) situation and lose all
survival pressure - so `bubble_factor` would never exceed 1.

Input : {state}
Output: IcmOutput
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.domain.poker import bubble_factor, icm, risk_premium


def _clip(lo: float, hi: float, x: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class IcmOutput:
    icm_ev: dict = field(default_factory=dict)
    hero_icm_ev: float = 0.0
    hero_equity_share: float = 0.0
    bubble_factor: dict = field(default_factory=dict)
    risk_premium: dict = field(default_factory=dict)
    pay_jump: dict = field(default_factory=dict)
    method: str = "chip_share"
    samples: int = 0
    chip_ev_note: str = ""
    confidence: float = 0.5
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class IcmAgent(BaseAgent):
    name: ClassVar[str] = "icm"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "ICM Agent"
    layer: ClassVar[str] = "tournament"
    depends: ClassVar[list[str]] = ["state"]
    supported_modes: ClassVar[list[str]] = ["tournament", "bounty"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = IcmOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        state = raw_input.get("state") or {}
        if not state:
            return False, "no state"
        if state.get("game_type") == "cash":
            return False, "cash game has no ICM"
        return True, ""

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _alive_stacks(state: dict) -> dict:
        """{pid: stack} for every player still alive in the tournament (not 'out')."""
        stacks: dict[str, float] = {}
        for p in state.get("players", []) or []:
            if p.get("status") == "out":
                continue
            pid = str(p.get("player_id"))
            s = float(p.get("stack") or 0)
            if s > 0:
                stacks[pid] = s
        return stacks

    @staticmethod
    def _payouts(tournament: dict | None) -> list:
        if not isinstance(tournament, dict):
            return []
        pay = tournament.get("payouts") or []
        return [float(x) for x in pay if isinstance(x, (int, float))]

    # ------------------------------------------------------------------ handle
    async def handle(self, inp: dict, ctx: AgentContext) -> IcmOutput:
        state = inp.get("state") or {}
        out = IcmOutput()
        out.chip_ev_note = "chip EV ≠ $EV：锦标赛里每一枚筹码的美元价值随名次与奖池非线性变化，ICM 把筹码份额折算为期望奖金。"

        tournament = state.get("tournament")
        stacks = self._alive_stacks(state)
        hero_id = state.get("hero_id")
        payouts = self._payouts(tournament)

        if not stacks:
            out.warnings.append({"code": "NO_STACKS", "message": "no alive players with stacks"})
            return out
        total = sum(stacks.values())
        if hero_id and hero_id in stacks:
            out.hero_equity_share = round(stacks[hero_id] / total, 6) if total > 0 else 0.0

        # ---- no payouts -> degrade to chip share ----
        if not payouts:
            out.warnings.append({"code": "NO_PAYOUTS", "message": "no payout structure; degrading to chip share"})
            out.method = "chip_share"
            out.icm_ev = {pid: round(s / total, 6) for pid, s in stacks.items()} if total > 0 else {}
            out.hero_icm_ev = out.icm_ev.get(hero_id, 0.0) if hero_id else 0.0
            out.confidence = 0.5
            out.evidence.append({"kind": "icm", "text": "chip-share fallback (no payouts)",
                                 "source": "icm", "weight": 0.5})
            return out

        # ---- real ICM ----
        cfg = ctx.config.icm_config()
        result = icm(stacks, payouts,
                     max_exact=int(cfg.get("max_exact", 9)),
                     samples=int(cfg.get("samples", 20000)),
                     seed=int(cfg.get("seed", 42)))
        out.method = result.method
        out.samples = result.samples
        out.icm_ev = {pid: round(v, 6) for pid, v in result.ev.items()}
        out.hero_icm_ev = out.icm_ev.get(hero_id, 0.0) if hero_id else 0.0

        # ---- bubble factor + risk premium for each other alive player ----
        if hero_id and hero_id in stacks:
            for pid in stacks:
                if pid == hero_id:
                    continue
                try:
                    bf = bubble_factor(stacks, payouts, hero_id, pid, max_exact=int(cfg.get("max_exact", 9)))
                    out.bubble_factor[pid] = round(bf, 6)
                    out.risk_premium[pid] = round(risk_premium(bf), 6)
                except Exception:
                    out.bubble_factor[pid] = 1.0
                    out.risk_premium[pid] = 0.0

        # ---- pay jump (approximation, documented) ----
        out.pay_jump = self._pay_jump(tournament, state, payouts)

        out.confidence = 1.0 if result.method == "exact" else 0.95
        out.evidence.append({"kind": "icm", "text":
                             f"ICM method={result.method} hero_icm_ev={out.hero_icm_ev} "
                             f"share={out.hero_equity_share}", "source": "icm", "weight": 1.0})
        return out

    @staticmethod
    def _pay_jump(tournament: dict | None, state: dict, payouts: list) -> dict:
        """Approximate current / next payout tiers.

        Approximation: the prize a player collects when they bust now is the
        payout for position `players_remaining` (when that position is in the
        money, i.e. players_remaining <= itm_players); otherwise they are not
        yet in the money (current_payout = 0).  `next_payout` is the next pay
        jump amount read from the state (smallest money prize while on/near the
        bubble).
        """
        if not isinstance(tournament, dict):
            return {"next_payout": None, "current_payout": None, "gap": None}
        next_payout = tournament.get("next_payout")
        next_payout = float(next_payout) if isinstance(next_payout, (int, float)) else None
        itm_players = int(tournament.get("itm_players") or len(payouts))
        remaining = int(tournament.get("players_remaining") or 0)
        current_payout = None
        if remaining > 0 and itm_players > 0 and remaining <= itm_players and payouts:
            idx = min(remaining - 1, len(payouts) - 1)
            current_payout = payouts[idx]
        else:
            current_payout = 0.0        # not yet in the money
        gap = (next_payout - current_payout) if (next_payout is not None and current_payout is not None) else None
        return {
            "next_payout": next_payout,
            "current_payout": current_payout,
            "gap": gap,
        }
