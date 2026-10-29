"""Emotion Agent (情绪/Tilt): deterministic tilt & overconfidence heuristics.

Probabilistic output (NOT fact). Reads the most recent normalized hand records
for hero and each active villain from ctx.repos.hands and maps losing streaks,
bad beats, big-pot losses, winning streaks, long folding spells and short-stack
pressure to tilt / overconfidence probabilities.

Input : {state, opponent, statistics}
Output: EmotionOutput
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext

_RECENT_N = 50
_VILLAIN_MIN_SAMPLES = 5


def _clip(lo: float, hi: float, x: float) -> float:
    return max(lo, min(hi, x))


def _num(v: object) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


@dataclass
class EmotionOutput:
    players: dict = field(default_factory=dict)
    hero: dict = field(default_factory=dict)
    confidence: float = 0.5
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class EmotionAgent(BaseAgent):
    name: ClassVar[str] = "emotion"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "情绪/Tilt Agent"
    layer: ClassVar[str] = "human"
    depends: ClassVar[list[str]] = ["state", "statistics"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = EmotionOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        if not raw_input.get("state"):
            return False, "no state"
        return True, ""

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _won(r: dict) -> bool | None:
        if isinstance(r.get("won_hand"), bool):
            return r.get("won_hand")
        nb = _num(r.get("net_bb"))
        if nb is not None:
            return nb > 0
        return None

    @staticmethod
    def _is_bad_beat(r: dict) -> bool:
        if r.get("bad_beat") is True:
            return True
        eq = _num(r.get("equity_at_decision"))
        if eq is not None and eq > 0.7:
            return EmotionAgent._won(r) is False
        return False

    @staticmethod
    def _sort_recent(records: list) -> list:
        """Newest-first; sorts by numeric ts when available, else insertion order."""
        if records and all(_num(r.get("ts")) is not None for r in records):
            return sorted(records, key=lambda r: _num(r.get("ts")) or 0.0, reverse=True)
        return list(reversed(records))

    def _records_for(self, player_id: str | None) -> list:
        if not player_id:
            return []
        try:
            hands = self._ctx_repos.hands.list(limit=10000, offset=0)
        except Exception:
            return []
        return [h for h in hands if isinstance(h, dict) and h.get("player_id") == player_id]

    def _compute(self, records: list, stack_bb: float) -> tuple[float, float, list[str], float]:
        """Return (tilt_prob, overconfidence_prob, evidence, confidence)."""
        if not records:
            # no hand data -> 0.5 neutral baseline, low confidence
            return 0.5, 0.5, [], 0.1

        recs = self._sort_recent(records)[:_RECENT_N]
        ev: list[str] = []
        tilt = 0.0
        over = 0.0

        # losing streak: last 5 all lost -> +0.3
        last5 = recs[:5]
        if len(last5) >= 5 and all(self._won(r) is False for r in last5):
            tilt += 0.3
            ev.append("5连败")
        # 8 of last 10 lost -> +0.2
        last10 = recs[:10]
        if len(last10) >= 10 and sum(1 for r in last10 if self._won(r) is False) >= 8:
            tilt += 0.2
            ev.append("近10手8败")

        # bad beat -> +0.3 (equity_at_decision > 0.7 and lost)
        if any(self._is_bad_beat(r) for r in recs):
            tilt += 0.3
            ev.append("bad beat")

        # big-pot loss -> +0.2 (net_bb < -50)
        if any((nb := _num(r.get("net_bb"))) is not None and nb < -50 for r in recs):
            tilt += 0.2
            ev.append("大底池亏损")

        # consecutive wins >= 5 -> overconfidence +0.25
        if len(last5) >= 5 and all(self._won(r) is True for r in last5):
            over += 0.25
            ev.append("5连胜")

        # long folding (last 20 vpip < 0.05) -> tilt +0.15 (boredom/impulse)
        recent20 = recs[:20]
        known = [r for r in recent20 if isinstance(r.get("vpip"), bool)]
        if len(known) >= 15 and sum(1 for r in known if r["vpip"]) / len(known) < 0.05:
            tilt += 0.15
            ev.append("长时间弃牌")

        # chip pressure -> tilt +0.15 (stack_bb < 20)
        if stack_bb > 0 and stack_bb < 20:
            tilt += 0.15
            ev.append("短码压力")

        tilt_prob = _clip(0.1, 0.9, 0.2 + tilt)
        over_prob = _clip(0.1, 0.9, 0.2 + over)
        confidence = _clip(0.1, 1.0, 0.3 + len(recs) / 200.0)
        return round(tilt_prob, 4), round(over_prob, 4), ev, round(confidence, 4)

    def _stack_bb(self, state: dict, pid: str | None) -> float:
        bb = float(state.get("bb") or 0) or 1.0
        for p in state.get("players", []) or []:
            if p.get("player_id") == pid:
                return float(p.get("stack") or 0) / bb
        return 0.0

    # ------------------------------------------------------------------ handle
    async def handle(self, inp: dict, ctx: AgentContext) -> EmotionOutput:
        self._ctx_repos = ctx.repos
        state = inp.get("state") or {}
        out = EmotionOutput()

        hero_id = state.get("hero_id")
        hero_records = self._records_for(hero_id)
        hero_stack_bb = self._stack_bb(state, hero_id)
        tilt, over, ev, conf = self._compute(hero_records, hero_stack_bb)
        out.hero = {
            "tilt_probability": tilt,
            "overconfidence_probability": over,
            "confidence": conf,
            "evidence": ev,
        }
        if not hero_records:
            out.warnings.append({"code": "NO_HERO_HANDS", "message": "no hand records for hero; 0.5 baseline"})

        # ---- each active villain ----
        for p in state.get("players", []) or []:
            if p.get("status") not in ("active", "allin"):
                continue
            pid = p.get("player_id")
            if pid == hero_id:
                continue
            recs = self._records_for(pid)
            if len(recs) < _VILLAIN_MIN_SAMPLES:
                out.players[str(pid)] = {
                    "tilt_probability": 0.5,
                    "overconfidence_probability": 0.5,
                    "confidence": 0.1,
                    "evidence": [],
                }
                continue
            t2, o2, ev2, c2 = self._compute(recs, self._stack_bb(state, pid))
            out.players[str(pid)] = {
                "tilt_probability": t2,
                "overconfidence_probability": o2,
                "confidence": c2,
                "evidence": ev2,
            }

        # overall confidence = hero confidence (primary subject)
        out.confidence = out.hero.get("confidence", 0.5)
        out.evidence.append({"kind": "emotion", "text":
                             f"hero tilt={out.hero['tilt_probability']} "
                             f"overconfidence={out.hero['overconfidence_probability']}",
                             "source": "hands", "weight": 1.0})
        return out
