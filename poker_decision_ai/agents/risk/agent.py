"""Risk Agent (风险): deterministic risk decomposition (never LLM-guessed).

Formulas (documented inline):
  - variance_bb = equity * (1 - equity)   (single-hand variance approximation;
    the max is 0.25 at equity=0.5).
  - bankroll_risk: 0.3 baseline; +0.2 for a big pot (pot_bb > 20 and
    to_call_bb > 10); +0.1 for a short hero stack (<40bb); clipped to [0,1].
  - tournament_risk: 1 - min(1, hero_icm_ev / equity_share)  (survival pressure:
    $EV below chip-equity share means ICM pressure).  null without icm input.
  - icm_risk: max(risk_premium) clipped to [0,1]; null without icm input.
  - decision_risk = 0.4 * variance_norm + 0.3 * stake_factor + 0.3 * icm_risk
    where stake_factor = to_call_bb / (to_call_bb + pot_bb), variance_norm =
    variance_bb / 0.25, and icm_risk defaults to 0 when absent.
  - high_risk_decision = decision_risk > 0.6 or (to_call_bb > 30 and equity < 0.4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext


@dataclass
class RiskOutput:
    variance_bb: float = 0.0
    bankroll_risk: float = 0.3
    tournament_risk: float | None = None
    icm_risk: float | None = None
    decision_risk: float = 0.0
    high_risk_decision: bool = False
    risk_factors: list = field(default_factory=list)
    confidence: float = 0.0
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _find_hero(players: list) -> dict | None:
    for p in players:
        if isinstance(p, dict) and p.get("is_hero"):
            return p
    return None


class RiskAgent(BaseAgent):
    name: ClassVar[str] = "risk"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "风险 Agent"
    layer: ClassVar[str] = "analysis"
    depends: ClassVar[list[str]] = ["state", "equity", "icm"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = RiskOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        if not raw_input.get("state"):
            return False, "state input is required for risk analysis"
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> RiskOutput:
        state = inp.get("state") or {}
        equity_in = inp.get("equity") or {}
        icm_in = inp.get("icm") or {}
        out = RiskOutput()
        warnings: list[dict] = []

        bb = float(state.get("bb") or 0.0) or 1.0
        pot = float(state.get("pot") or 0.0)
        to_call = float(state.get("to_call") or 0.0)
        pot_bb = pot / bb
        to_call_bb = to_call / bb

        hero_equity = equity_in.get("hero_equity", 0.5)
        if hero_equity is None:
            warnings.append({"code": "NO_EQUITY",
                             "message": "equity input missing; using neutral 0.5"})
            hero_equity = 0.5
        hero_equity = float(hero_equity)

        # ---- single-hand variance approximation p*(1-p) ----
        out.variance_bb = round(hero_equity * (1 - hero_equity), 4)

        # ---- bankroll risk ----
        bankroll = 0.3
        if pot_bb > 20 and to_call_bb > 10:
            bankroll += 0.2
        hero_p = _find_hero(state.get("players", []) or [])
        stack_bb = float(hero_p.get("stack", 0.0) or 0.0) / bb if hero_p else None
        if stack_bb is not None and stack_bb < 40:
            bankroll += 0.1
        out.bankroll_risk = round(min(1.0, max(0.0, bankroll)), 3)

        # ---- tournament / ICM risk ----
        if icm_in:
            hero_icm_ev = icm_in.get("hero_icm_ev")
            equity_share = icm_in.get("hero_equity_share")
            if hero_icm_ev is not None and equity_share:
                share = max(float(equity_share), 1e-9)
                out.tournament_risk = round(
                    max(0.0, min(1.0, 1 - min(1.0, float(hero_icm_ev) / share))), 4)
            else:
                out.tournament_risk = 0.0

            rp = icm_in.get("risk_premium") or {}
            vals: list[float] = []
            if isinstance(rp, dict):
                vals = [float(v) for v in rp.values() if isinstance(v, (int, float))]
            elif isinstance(rp, (int, float)):
                vals = [float(rp)]
            out.icm_risk = round(min(1.0, max(0.0, max(vals))), 4) if vals else 0.0
        else:
            out.tournament_risk = None
            out.icm_risk = None

        # ---- decision risk ----
        variance_norm = out.variance_bb / 0.25            # max single-hand variance
        stake_factor = to_call_bb / (to_call_bb + pot_bb) if (to_call_bb + pot_bb) > 0 else 0.0
        icm_risk = out.icm_risk if out.icm_risk is not None else 0.0
        out.decision_risk = round(
            0.4 * min(1.0, variance_norm) + 0.3 * min(1.0, stake_factor) + 0.3 * icm_risk, 4)

        out.high_risk_decision = (out.decision_risk > 0.6) or (to_call_bb > 30 and hero_equity < 0.4)

        # ---- human-readable risk factors ----
        factors: list[str] = []
        if pot_bb > 20 and to_call_bb > 10:
            factors.append("大底池高注码")
        if stack_bb is not None and stack_bb < 40:
            factors.append("hero 短码 (<40bb)")
        if out.tournament_risk is not None and out.tournament_risk > 0.3:
            factors.append("锦标赛生存压力")
        if out.icm_risk is not None and out.icm_risk > 0.2:
            factors.append("ICM 风险溢价")
        if stake_factor > 0.5:
            factors.append("高注码比例")
        if hero_equity < 0.4:
            factors.append("低 equity")
        out.risk_factors = factors

        out.confidence = 1.0  # fully deterministic formulas
        out.evidence = [{"kind": "risk",
                         "text": f"decision_risk={out.decision_risk}, "
                                 f"high_risk={out.high_risk_decision}, factors={factors}",
                         "source": "risk", "weight": 1.0}]
        out.warnings = warnings
        return out
