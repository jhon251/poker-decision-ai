"""Strategy Evolution Agent (策略演化): detect drift / sudden shift in a player.

Deterministic. For each active villain with hand records, split records into
historical (all) / recent (last 20) / older (earliest 100 before the recent
window), then compare vpip / pfr / aggression_frequency via the domain stats
engine.  Thresholds:
  - drift:        |recent - historical| > 0.15 on vpip or aggression_frequency
  - sudden_shift: |recent - older|     > 0.25 on vpip or aggression_frequency
  - change_score: min(1, max_abs_diff / 0.3)

Input : {state, opponent}
Output: StrategyEvolutionOutput
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.domain.poker import aggression_frequency, vpip


def _clip(lo: float, hi: float, x: float) -> float:
    return max(lo, min(hi, x))


def _num(v: object) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _sort_chrono(records: list) -> list:
    """Oldest-first. Uses numeric ts when all present, else insertion order."""
    if records and all(_num(r.get("ts")) is not None for r in records):
        return sorted(records, key=lambda r: _num(r.get("ts")) or 0.0)
    return list(records)


def _regime_label(vpip_val: float | None, agg_val: float | None) -> str:
    """Simplified human_style-style mapping from vpip / aggression thresholds."""
    v = vpip_val if vpip_val is not None else 0.25
    a = agg_val if agg_val is not None else 0.35
    if v <= 0.15 and a <= 0.3:
        return "Nit"
    if v <= 0.25 and a >= 0.5:
        return "TAG"
    if v >= 0.35 and a >= 0.55:
        return "LAG"
    if v >= 0.45 and a >= 0.7:
        return "Maniac"
    return "Regular"


@dataclass
class StrategyEvolutionOutput:
    players: dict = field(default_factory=dict)
    confidence: float = 0.5
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class StrategyEvolutionAgent(BaseAgent):
    name: ClassVar[str] = "strategy_evolution"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "策略演化 Agent"
    layer: ClassVar[str] = "learning"
    depends: ClassVar[list[str]] = ["state", "opponent"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = StrategyEvolutionOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        if not raw_input.get("state"):
            return False, "no state"
        return True, ""

    # ------------------------------------------------------------------ helpers
    def _records_for(self, player_id: str | None, ctx: AgentContext) -> list:
        if not player_id:
            return []
        try:
            hands = ctx.repos.hands.list(limit=10000, offset=0)
        except Exception:
            return []
        return [h for h in hands if isinstance(h, dict) and h.get("player_id") == player_id]

    @staticmethod
    def _metric_pair(recs: list):
        """Return (vpip_value, agg_value, vpip_samples, agg_samples)."""
        v = vpip(recs)
        a = aggression_frequency(recs)
        return (v.value if v.samples else None,
                a.value if a.samples else None,
                v.samples, a.samples)

    # ------------------------------------------------------------------ handle
    async def handle(self, inp: dict, ctx: AgentContext) -> StrategyEvolutionOutput:
        state = inp.get("state") or {}
        opponent = inp.get("opponent") if isinstance(inp.get("opponent"), dict) else None

        out = StrategyEvolutionOutput()
        total_conf = 0.0
        n_players = 0

        for p in state.get("players", []) or []:
            if p.get("status") not in ("active", "allin"):
                continue
            pid = p.get("player_id")
            if pid == state.get("hero_id"):
                continue
            n_players += 1
            records = self._records_for(pid, ctx)
            if not records:
                out.players[str(pid)] = {
                    "regime_change": False,
                    "drift_detected": False,
                    "sudden_shift": False,
                    "adaptation_direction": "stable",
                    "current_regime": "unknown",
                    "previous_regime": "unknown",
                    "change_score": 0.0,
                    "evidence": [],
                }
                out.warnings.append({"code": "NO_HANDS", "message": f"no hand records for {pid}"})
                continue

            chrono = _sort_chrono(records)
            historical = chrono
            recent = chrono[-20:]
            older = chrono[:-20][:100]          # earliest 100 before the recent window

            rv, ra, rvs, ras = self._metric_pair(recent)
            hv, ha, hvs, has = self._metric_pair(historical)
            ov, oa, ovs, oas = self._metric_pair(older)

            d_v_hist = abs(rv - hv) if (rv is not None and hv is not None) else 0.0
            d_a_hist = abs(ra - ha) if (ra is not None and ha is not None) else 0.0
            d_v_older = abs(rv - ov) if (rv is not None and ov is not None) else 0.0
            d_a_older = abs(ra - oa) if (ra is not None and oa is not None) else 0.0

            drift = d_v_hist > 0.15 or d_a_hist > 0.15
            sudden = d_v_older > 0.25 or d_a_older > 0.25

            # direction: the axis (vpip or aggression) with the largest recent-vs-
            # historical change decides adaptation_direction and its sign.
            axis_deltas = {
                "vpip": (rv - hv) if (rv is not None and hv is not None) else 0.0,
                "aggression": (ra - ha) if (ra is not None and ha is not None) else 0.0,
            }
            axis = max(axis_deltas, key=lambda k: abs(axis_deltas[k]))
            mag = axis_deltas[axis]
            if abs(mag) < 0.05:
                direction = "stable"
            elif axis == "vpip":
                direction = "looser" if mag > 0 else "tighter"
            else:
                direction = "more_aggressive" if mag > 0 else "more_passive"

            # regime_change = drift AND direction consistent with the historical
            # trend (recent-vs-historical sign matches recent-vs-older sign).
            s_hist = 1.0 if mag > 0 else (-1.0 if mag < 0 else 0.0)
            if axis == "vpip":
                trend = (rv - ov) if (rv is not None and ov is not None) else 0.0
            else:
                trend = (ra - oa) if (ra is not None and oa is not None) else 0.0
            s_older = 1.0 if trend > 0 else (-1.0 if trend < 0 else 0.0)
            consistent = s_hist != 0.0 and s_hist == s_older
            regime_change = drift and consistent

            change_score = _clip(0.0, 1.0, max(d_v_hist, d_a_hist, d_v_older, d_a_older) / 0.3)

            # regime labels: prefer opponent.strategy_regime.label for current
            cur_label = None
            if isinstance(opponent, dict):
                ops = opponent.get("players")
                if isinstance(ops, dict) and isinstance(ops.get(pid), dict):
                    sr = ops[pid].get("strategy_regime")
                    if isinstance(sr, dict) and sr.get("label"):
                        cur_label = str(sr["label"])
            current_regime = cur_label or _regime_label(rv, ra)
            previous_regime = _regime_label(ov, oa)

            out.players[str(pid)] = {
                "regime_change": bool(regime_change),
                "drift_detected": bool(drift),
                "sudden_shift": bool(sudden),
                "adaptation_direction": direction,
                "current_regime": current_regime,
                "previous_regime": previous_regime,
                "change_score": round(change_score, 4),
                "evidence": [
                    f"vpip recent={rv} hist={hv} older={ov}",
                    f"agg recent={ra} hist={ha} older={oa}",
                    f"drift={drift} sudden={sudden} direction={direction}",
                ],
            }
            total_conf += _clip(0.1, 1.0, 0.3 + len(records) / 500.0)

        out.confidence = round(total_conf / n_players, 4) if n_players else 0.2
        if n_players == 0:
            out.warnings.append({"code": "NO_VILLAINS", "message": "no active villains in state"})
        return out
