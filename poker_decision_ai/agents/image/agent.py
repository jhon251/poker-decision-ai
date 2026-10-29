"""Image Agent (桌面形象): perceived table image of hero + villain reactions.

Deterministic. Compares hero stats against same-segment population means to
derive relative image scores, then produces rule-based reactions for each
active villain from their archetype.

Input : {state, statistics, population, opponent}
Output: ImageOutput
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext

_NEUTRAL_VPIP = 0.25      # documented default when hero vpip is unknown
_NEUTRAL_AGG = 0.35       # documented default when aggression is unknown
_NEUTRAL_FOLD_TO_CBET = 0.55


def _clip(lo: float, hi: float, x: float) -> float:
    return max(lo, min(hi, x))


def _metric_value(metric: object) -> float | None:
    """Read {value, samples} or {value: float} metric dicts."""
    if isinstance(metric, dict) and isinstance(metric.get("value"), (int, float)):
        return float(metric["value"])
    if isinstance(metric, (int, float)):
        return float(metric)
    return None


def _metric_samples(metric: object) -> int:
    if isinstance(metric, dict):
        return int(metric.get("samples") or 0)
    return 0


@dataclass
class ImageOutput:
    hero_image: dict = field(default_factory=dict)
    villain_reactions: dict = field(default_factory=dict)
    confidence: float = 0.5
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class ImageAgent(BaseAgent):
    name: ClassVar[str] = "image"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "桌面形象 Agent"
    layer: ClassVar[str] = "human"
    depends: ClassVar[list[str]] = ["state", "statistics", "population", "opponent"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = ImageOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        if not raw_input.get("state"):
            return False, "no state"
        return True, ""

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _relative_score(hero_val: float, baseline: float) -> float:
        """Approximate 'percentile': 0.5 when equal to baseline, saturating 0..1.

        score = clip01(hero_val / (2 * baseline)).  hero_val == baseline -> 0.5;
        hero_val == 2*baseline -> 1.0.  When baseline <= 0, fall back to raw value.
        """
        if baseline <= 0:
            return _clip(0.0, 1.0, hero_val)
        return _clip(0.0, 1.0, hero_val / (2.0 * baseline))

    @staticmethod
    def _archetype(fv: dict) -> str:
        tight = float(fv.get("tightness", 0.5))
        agg = float(fv.get("aggression", 0.5))
        if tight >= 0.75 and agg < 0.35:
            return "Nit"
        if tight >= 0.6 and agg >= 0.55:
            return "TAG"      # tight-aggressive
        if tight < 0.4 and agg >= 0.6:
            return "LAG"
        return "Regular"

    # ------------------------------------------------------------------ handle
    async def handle(self, inp: dict, ctx: AgentContext) -> ImageOutput:
        state = inp.get("state") or {}
        statistics = inp.get("statistics") if isinstance(inp.get("statistics"), dict) else None
        population = inp.get("population") if isinstance(inp.get("population"), dict) else None
        opponent = inp.get("opponent") if isinstance(inp.get("opponent"), dict) else None

        out = ImageOutput()

        # ---- hero stats ----
        hero_stats = (statistics or {}).get("hero_stats") if statistics else None
        if not isinstance(hero_stats, dict):
            hero_stats = {}

        hero_vpip = _metric_value(hero_stats.get("vpip"))
        hero_agg = _metric_value(hero_stats.get("aggression_frequency"))
        hero_f2c = _metric_value(hero_stats.get("fold_to_cbet"))
        vpip_samples = _metric_samples(hero_stats.get("vpip"))
        agg_samples = _metric_samples(hero_stats.get("aggression_frequency"))

        perceived_vpip = hero_vpip if hero_vpip is not None else _NEUTRAL_VPIP
        perceived_agg = hero_agg if hero_agg is not None else _NEUTRAL_AGG

        # ---- population baseline (same-segment mean when available) ----
        pop_stats = population.get("stats") if population else None
        if not isinstance(pop_stats, dict):
            pop_stats = {}
        base_vpip = _metric_value(pop_stats.get("vpip")) or _NEUTRAL_VPIP
        base_agg = _metric_value(pop_stats.get("aggression_frequency")) or _NEUTRAL_AGG

        # ---- scores ----
        loose = self._relative_score(perceived_vpip, base_vpip)
        tight = 1.0 - loose
        aggressive = self._relative_score(perceived_agg, base_agg)
        passive = 1.0 - aggressive
        frequent_bluffer = aggressive * 0.6          # spec: aggression*0.6
        if hero_f2c is not None:
            # overfolder = how far fold_to_cbet sits above 0.35 on a 0.5-wide ramp
            overfolder = _clip(0.0, 1.0, (hero_f2c - 0.35) / 0.5)
        else:
            overfolder = 0.5

        scores = {
            "tight": round(tight, 4),
            "loose": round(loose, 4),
            "aggressive": round(aggressive, 4),
            "passive": round(passive, 4),
            "frequent_bluffer": round(frequent_bluffer, 4),
            "overfolder": round(overfolder, 4),
        }
        labels = [name for name in ("Tight", "Loose", "Aggressive", "Passive",
                                    "Frequent_Bluffer", "Overfolder")
                  if scores[name.lower()] > 0.6]

        # sample-poor bluffer estimate -> low confidence flag
        low_sample = min(vpip_samples, agg_samples) < 20 if (vpip_samples or agg_samples) else True
        if low_sample:
            out.warnings.append({"code": "LOW_SAMPLE", "message":
                                 "hero sample size small; image scores shrink toward neutral"})

        out.hero_image = {
            "labels": labels,
            "scores": scores,
            "perceived_vpip": round(perceived_vpip, 4),
            "perceived_aggression": round(perceived_agg, 4),
            "bluffer_frequency": round(frequent_bluffer, 4),
        }

        # ---- villain reactions ----
        hero_aggressive_image = aggressive > 0.6
        for p in state.get("players", []) or []:
            if p.get("status") not in ("active", "allin"):
                continue
            pid = p.get("player_id")
            if pid == state.get("hero_id"):
                continue
            prof = {}
            if isinstance(opponent, dict):
                ops = opponent.get("players")
                if isinstance(ops, dict) and isinstance(ops.get(pid), dict):
                    prof = ops[pid]
            fv = prof.get("feature_vector") if isinstance(prof.get("feature_vector"), dict) else {}
            if fv:
                arch = self._archetype(fv)
                source = "opponent"
            else:
                arch = "Regular"          # no opponent data -> population-informed default
                source = "population"
            out.villain_reactions[str(pid)] = {
                "adjustment": self._reaction_text(arch, hero_aggressive_image),
                "note": f"基于{source}画像({arch})与hero激进形象={hero_aggressive_image}的规则反应",
            }

        # ---- confidence ----
        n = max(vpip_samples, agg_samples, 0)
        out.confidence = round(_clip(0.2, 1.0, 0.3 + n / 200.0), 4)
        if statistics is None:
            out.confidence = min(out.confidence, 0.4)
            out.warnings.append({"code": "NO_STATISTICS", "message": "no statistics input; using defaults"})

        out.evidence.append({"kind": "image", "text":
                             f"hero perceived_vpip={perceived_vpip:.2f} agg={perceived_agg:.2f} "
                             f"labels={labels}", "source": "statistics", "weight": 1.0})
        return out

    @staticmethod
    def _reaction_text(arch: str, hero_aggressive: bool) -> str:
        if not hero_aggressive:
            return "标准应对"
        if arch == "TAG":
            return "更多慢打与抓诈"
        if arch == "LAG":
            return "更少尊重下注"
        if arch == "Nit":
            return "更多弃牌"
        return "标准应对"
