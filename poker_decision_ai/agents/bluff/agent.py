"""Bluff Agent (诈唬分析): classify the opponent's bet as value vs bluff.

Deterministic Bayesian-style synthesis. No LLM is used for the numeric
probabilities (spec: "确定性优先"); reasoning is a deterministic template.

Input : {state, board, range, equity, solver, opponent, population,
         confidence, strategy_evolution, emotion, icm, tournament_stage}
Output: BluffOutput (value_probability + bluff_probability sum to exactly 1).

Key assumption (documented inline): we do NOT use the naive rule
"big bet = bluff". An overbet against a balanced player is POLARIZED
(value AND bluff), so its prior is only 0.5 and deviations come from
player profile / history / emotion / board texture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext


def _clip(lo: float, hi: float, x: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class BluffOutput:
    value_probability: float = 0.5
    bluff_probability: float = 0.5
    confidence: float = 0.5
    reasoning: str = ""
    implied_action: str = "fold"
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class BluffAgent(BaseAgent):
    name: ClassVar[str] = "bluff"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "诈唬分析 Agent"
    layer: ClassVar[str] = "analysis"
    depends: ClassVar[list[str]] = ["state", "opponent", "population", "board",
                                    "emotion", "strategy_evolution", "icm"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = BluffOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        state = raw_input.get("state") or {}
        if not state:
            return False, "no state"
        if float(state.get("current_bet") or 0) <= 0:
            return False, "no opponent bet to classify (current_bet <= 0)"
        return True, ""

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _bettor(state: dict) -> str | None:
        """The opponent whose bet we classify (last aggressor unless it is hero)."""
        hero_id = state.get("hero_id")
        la = state.get("last_aggressor")
        if la and la != hero_id:
            return str(la)
        for p in state.get("players", []) or []:
            pid = p.get("player_id")
            if pid != hero_id and p.get("status") in ("active", "allin"):
                return str(pid)
        return None

    @staticmethod
    def _bet_size_ratio(state: dict) -> float:
        """Bet size as a fraction of the pot BEFORE the bet (poker convention).

        ratio = current_bet / (pot - current_bet).  Falls back to
        current_bet / pot when the stored `pot` is already net of the bet.
        """
        pot = float(state.get("pot") or 0)
        bet = float(state.get("current_bet") or 0)
        if pot <= 0 or bet <= 0:
            return 0.0
        pot_before = pot - bet
        if pot_before > 1e-9:
            return bet / pot_before
        return bet / pot

    @staticmethod
    def _gto_bluff_prior(ratio: float) -> float:
        """GTO baseline bluff frequency by bet sizing (comment: spec §4 bluff).

        Small bets are value-heavy; as size grows the range polarizes toward a
        50/50 value|bluff split on overbets. These are heuristics, not solved
        GTO - they serve as the Bayesian prior only.
        """
        if ratio <= 0.5:
            return 0.2          # ~0.33 pot
        if ratio <= 0.85:
            return 0.33         # ~0.66 pot
        if ratio <= 1.2:
            return 0.4          # ~1.0 pot
        return 0.5              # overbet: polarized (value & bluff both poles)

    @staticmethod
    def _population_bluff_prior(population: dict | None) -> tuple[float | None, int]:
        """Extract a population bluff/value frequency for the street, if present.

        Searches population.stats and population.segments[primary_segment] for
        metric names containing 'bluff' (direct bluff frequency) or 'value'
        (value frequency -> bluff = 1 - value).
        """
        if not isinstance(population, dict):
            return None, 0

        def search(container) -> tuple[float | None, int]:
            if not isinstance(container, dict):
                return None, 0
            for key, v in container.items():
                if not isinstance(v, dict) or "value" not in v:
                    continue
                kl = str(key).lower()
                val = v.get("value")
                if not isinstance(val, (int, float)):
                    continue
                samples = int(v.get("samples") or 0)
                if "bluff" in kl:
                    return _clip(0.0, 1.0, float(val)), samples
                if "value" in kl and "value_bet" not in kl:
                    return _clip(0.0, 1.0, 1.0 - float(val)), samples
            return None, 0

        found, n = search(population.get("stats"))
        if found is not None:
            return found, n
        primary = population.get("primary_segment")
        segments = population.get("segments")
        if isinstance(segments, dict) and primary:
            found, n = search(segments.get(primary))
            if found is not None:
                return found, n
        return None, 0

    @staticmethod
    def _player_profile(opponent: dict | None, pid: str | None) -> dict:
        if not isinstance(opponent, dict) or not pid:
            return {}
        players = opponent.get("players")
        if not isinstance(players, dict):
            return {}
        prof = players.get(pid)
        return prof if isinstance(prof, dict) else {}

    # ------------------------------------------------------------------ handle
    async def handle(self, inp: dict, ctx: AgentContext) -> BluffOutput:
        state = inp.get("state") or {}
        board = inp.get("board") if isinstance(inp.get("board"), dict) else None
        equity = inp.get("equity") if isinstance(inp.get("equity"), dict) else None
        opponent = inp.get("opponent") if isinstance(inp.get("opponent"), dict) else None
        population = inp.get("population") if isinstance(inp.get("population"), dict) else None
        emotion = inp.get("emotion") if isinstance(inp.get("emotion"), dict) else None
        se = inp.get("strategy_evolution") if isinstance(inp.get("strategy_evolution"), dict) else None
        icm = inp.get("icm") if isinstance(inp.get("icm"), dict) else None

        out = BluffOutput()
        street = str(state.get("street", ""))
        bettor = self._bettor(state)
        ratio = self._bet_size_ratio(state)

        if bettor is None:
            out.warnings.append({"code": "NO_BETTOR", "message": "could not identify the betting opponent"})
            out.reasoning = "无法识别下注对手，输出中性先验 0.5/0.5。"
            out.value_probability = 0.5
            out.bluff_probability = 0.5
            out.confidence = 0.2
            out.implied_action = "fold"
            return out

        # ---- 1) prior ----
        prior = self._gto_bluff_prior(ratio)
        pop_prior, pop_n = self._population_bluff_prior(population)
        if pop_prior is not None:
            prior = pop_prior
            out.evidence.append({"kind": "population", "text":
                                 f"群体先验 bluff={pop_prior:.2f} (samples={pop_n})", "source": "population", "weight": 1.0})
        else:
            out.evidence.append({"kind": "prior", "text":
                                 f"GTO 尺寸先验 bluff={prior:.2f} (bet/pot={ratio:.2f})", "source": "solver", "weight": 0.8})

        # ---- 2) adjustment multipliers (each in 0.5..1.5) ----
        profile = self._player_profile(opponent, bettor)
        fv = profile.get("feature_vector") if isinstance(profile.get("feature_vector"), dict) else {}
        bt = fv.get("bluff_tendency", 0.5)
        ct = fv.get("calling_tendency", 0.5)
        tightness = fv.get("tightness", 0.5)
        label = str((profile.get("strategy_regime") or {}).get("label", "") if isinstance(profile.get("strategy_regime"), dict) else "")
        sample_size = int(profile.get("sample_size") or 0)

        mults: list[tuple[str, float]] = []

        # bluff tendency high -> up (0.5 + bt in [0.5, 1.5]; neutral at 0.5)
        mults.append(("bluff_tendency", _clip(0.5, 1.5, 0.5 + float(bt))))

        # emotion tilt high -> up
        if isinstance(emotion, dict):
            players = emotion.get("players")
            if isinstance(players, dict) and isinstance(players.get(bettor), dict):
                tilt = players[bettor].get("tilt_probability", 0.5)
                if isinstance(tilt, (int, float)):
                    mults.append(("tilt", _clip(0.5, 1.5, 0.5 + float(tilt))))

        # strategy evolution: recent more aggressive -> up; more passive -> down
        if isinstance(se, dict):
            se_players = se.get("players")
            if isinstance(se_players, dict) and isinstance(se_players.get(bettor), dict):
                direction = se_players[bettor].get("adaptation_direction", "stable")
                if direction == "more_aggressive":
                    mults.append(("strategy_evolution", 1.25))
                elif direction == "more_passive":
                    mults.append(("strategy_evolution", 0.8))

        # busted draw on the river -> up (missed flush/straight makes bluff more likely)
        if board is not None and street == "river":
            ds = board.get("draw_structure") or []
            if "flush_draw" in ds or "straight_draw" in ds:
                mults.append(("busted_draw", 1.25))

        # abnormally large bet (>1.5 pot) + high calling tendency -> up
        if ratio > 1.5 and float(ct) > 0.6:
            mults.append(("overbet_vs_caller", 1.25))

        # ICM pressure high -> down (afraid of elimination; value-heavy)
        if isinstance(icm, dict):
            rp_map = icm.get("risk_premium")
            rp = None
            if isinstance(rp_map, dict):
                rp = rp_map.get(bettor)
                if rp is None:
                    vals = [v for v in rp_map.values() if isinstance(v, (int, float))]
                    rp = max(vals) if vals else None
            if isinstance(rp, (int, float)) and rp > 0:
                mults.append(("icm_pressure", _clip(0.5, 1.0, 1.0 - 0.5 * float(rp))))

        # Nit/TAG -> big down
        if tightness > 0.65 or label in ("Nit", "TAG", "nit", "tag"):
            mults.append(("nit_tag", 0.6))

        # short stack -> slight down
        stack = 0.0
        for p in state.get("players", []) or []:
            if p.get("player_id") == bettor:
                stack = float(p.get("stack") or 0)
                break
        bb = float(state.get("bb") or 0) or 1.0
        stack_bb = stack / bb
        if stack_bb > 0 and stack_bb < 20:
            mults.append(("short_stack", 0.9))

        # ---- 3) synthesize ----
        product = prior
        applied: list[str] = []
        for name, m in mults:
            product *= m
            if abs(m - 1.0) > 1e-9:
                applied.append(f"{name}×{m:.2f}")
        bluff = _clip(0.02, 0.98, product)
        value = 1.0 - bluff
        out.bluff_probability = round(bluff, 4)
        out.value_probability = round(value, 4)

        # ---- 4) implied action for the Judge ----
        hero_eq = equity.get("hero_equity") if equity is not None else None
        if out.bluff_probability >= 0.55:
            out.implied_action = "call_or_raise"
        elif isinstance(hero_eq, (int, float)) and hero_eq >= 0.5:
            out.implied_action = "call"      # equity still supports continuing
        else:
            out.implied_action = "fold"

        # ---- 5) confidence + warnings ----
        conf = 0.5
        if sample_size >= 100:
            conf += 0.2
        elif sample_size >= 20:
            conf += 0.1
        if pop_prior is not None:
            conf += 0.1
        if opponent is None:
            conf -= 0.2
            out.warnings.append({"code": "NO_OPPONENT", "message": "no opponent profile; using size prior only"})
        if population is None:
            conf -= 0.1
            out.warnings.append({"code": "NO_POPULATION", "message": "no population data; using GTO size prior"})
        out.confidence = round(_clip(0.1, 1.0, conf), 4)

        if applied:
            out.evidence.append({"kind": "adjustments", "text":
                                 "调整因子: " + ", ".join(applied), "source": "bluff", "weight": 1.0})

        out.reasoning = (
            f"对手({bettor})在{street or '?'}下注 bet/pot={ratio:.2f}。"
            f"先验诈唬概率 {prior:.2f}，经画像/情绪/历史/牌面因子调整后 "
            f"bluff={out.bluff_probability:.2f}，value={out.value_probability:.2f}。"
        )
        if not applied:
            out.reasoning += " 无显著偏离因子，仅凭下注尺寸与先验判断（避免“下注大=诈唬”的简单规则）。"
        return out
