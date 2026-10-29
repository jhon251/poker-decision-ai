"""Deterministic poker statistics engine (spec §十一: never LLM-computed).

All metrics take a normalized hand record (see HAND_RECORD_FIELDS) and return
{value, samples} pairs so the Confidence Agent can weigh them correctly.
Bayesian shrinkage toward a prior and time-decay weighting are first-class.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

#: Canonical normalized hand record consumed by the statistics engine.
#: Produced by the Statistics Agent's `normalize_hand` from a PokerState
#: (+ outcome).  Hand-history/tracker adapters must map to this shape.
HAND_RECORD_FIELDS = [
    "hand_id", "player_id", "game", "stakes", "n_players", "position",
    "vpip", "pfr", "three_bet", "faced_3bet", "folded_to_3bet", "four_bet",
    "cbet_opportunity", "cbet", "faced_cbet", "folded_to_cbet",
    "saw_flop", "went_to_showdown", "won_at_showdown",
    "won_hand", "net_bb", "bets", "calls", "folds", "ts",
]


@dataclass
class Metric:
    value: float
    samples: int

    def to_dict(self) -> dict:
        return {"value": round(self.value, 4), "samples": self.samples}


@dataclass
class StatsBundle:
    metrics: dict = field(default_factory=dict)     # metric name -> Metric

    def to_dict(self) -> dict:
        return {k: v.to_dict() for k, v in sorted(self.metrics.items())}


def _rate(hands: list, field: str) -> Metric:
    n = sum(1 for h in hands if isinstance(h.get(field), bool))
    if n == 0:
        return Metric(0.0, 0)
    return Metric(sum(1 for h in hands if h.get(field)) / n, n)


def _opportunity_rate(hands: list, opp_field: str, act_field: str) -> Metric:
    """e.g. fold_to_cbet: only hands where the player faced a cbet count."""
    n = sum(1 for h in hands if h.get(opp_field))
    if n == 0:
        return Metric(0.0, 0)
    hits = sum(1 for h in hands if h.get(opp_field) and h.get(act_field))
    return Metric(hits / n, n)


def vpip(hands: list) -> Metric:
    return _rate(hands, "vpip")


def pfr(hands: list) -> Metric:
    return _rate(hands, "pfr")


def three_bet(hands: list) -> Metric:
    return _rate(hands, "three_bet")


def fold_to_3bet(hands: list) -> Metric:
    return _opportunity_rate(hands, "faced_3bet", "folded_to_3bet")


def four_bet(hands: list) -> Metric:
    return _opportunity_rate(hands, "faced_3bet", "four_bet")


def cbet(hands: list) -> Metric:
    return _opportunity_rate(hands, "cbet_opportunity", "cbet")


def fold_to_cbet(hands: list) -> Metric:
    return _opportunity_rate(hands, "faced_cbet", "folded_to_cbet")


def wtsd(hands: list) -> Metric:
    saw_flop = [h for h in hands if h.get("saw_flop")]
    if not saw_flop:
        return Metric(0.0, 0)
    return Metric(sum(1 for h in saw_flop if h.get("went_to_showdown")) / len(saw_flop), len(saw_flop))


def won_at_showdown(hands: list) -> Metric:
    sd = [h for h in hands if h.get("went_to_showdown")]
    if not sd:
        return Metric(0.0, 0)
    known = [h for h in sd if isinstance(h.get("won_at_showdown"), bool)]
    if not known:
        return Metric(0.0, 0)
    return Metric(sum(1 for h in known if h["won_at_showdown"]) / len(known), len(known))


def wwsf(hands: list) -> Metric:
    saw_flop = [h for h in hands if h.get("saw_flop")]
    if not saw_flop:
        return Metric(0.0, 0)
    return Metric(sum(1 for h in saw_flop if h.get("won_hand")) / len(saw_flop), len(saw_flop))


def aggression_factor(hands: list) -> Metric:
    bets = sum(int(h.get("bets") or 0) for h in hands)
    calls = sum(int(h.get("calls") or 0) for h in hands)
    if calls == 0:
        return Metric(float("inf") if bets > 0 else 0.0, len(hands))
    return Metric(bets / calls, len(hands))


def aggression_frequency(hands: list) -> Metric:
    """(bets+raises) / (bets+raises+calls+folds) across all opportunities."""
    total = 0
    agg = 0
    for h in hands:
        b = int(h.get("bets") or 0)
        c = int(h.get("calls") or 0)
        f = int(h.get("folds") or 0)
        if b + c + f == 0:
            continue
        agg += b
        total += b + c + f
    if total == 0:
        return Metric(0.0, 0)
    return Metric(agg / total, total)


def roi(hands: list) -> Metric:
    invested = sum(max(0.0, -float(h.get("net_bb") or 0.0)) for h in hands)
    won = sum(float(h.get("net_bb") or 0.0) for h in hands)
    if invested <= 0:
        return Metric(0.0, len(hands))
    return Metric(won / invested * 100.0, len(hands))


def itm(tournament_results: list) -> Metric:
    n = len(tournament_results)
    if n == 0:
        return Metric(0.0, 0)
    return Metric(sum(1 for r in tournament_results if r.get("in_the_money")) / n, n)


def bb_per_100(hands: list) -> Metric:
    if not hands:
        return Metric(0.0, 0)
    return Metric(sum(float(h.get("net_bb") or 0.0) for h in hands) / len(hands) * 100.0, len(hands))


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (deterministic confidence bounds)."""
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def time_decay_weights(timestamps: list, half_life_days: float = 90.0, now: float | None = None) -> list[float]:
    """Exponential time-decay weights (newest ~1). now defaults to max ts."""
    import time as _time

    if not timestamps:
        return []
    now = max(timestamps) if now is None else now
    lam = math.log(2) / max(half_life_days, 0.001)
    return [math.exp(-lam * max(0.0, now - t) / 86400.0) for t in timestamps]


def bayes_shrink(observed_rate: float, n: int, prior_rate: float, prior_strength: int = 200) -> float:
    """Shrink an observed rate toward a prior (population mean).

    weight = n / (n + prior_strength) - small samples lean on the prior.
    This is the mathematical backbone of the Population->Type->Individual
    three-layer model.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    weight = n / (n + prior_strength)
    return weight * observed_rate + (1 - weight) * prior_rate


def weighted_mean(hands: list, field: str, weights: list[float]) -> float:
    vals = [float(h[field]) for h in hands if h.get(field) is not None]
    if not vals:
        return 0.0
    ws = weights[: len(vals)] or [1.0] * len(vals)
    return sum(v * w for v, w in zip(vals, ws)) / sum(ws)


def compute_all(hands: list, tournament_results: list | None = None) -> StatsBundle:
    """Compute the full standard metric bundle over normalized hands."""
    b = StatsBundle()
    b.metrics.update({
        "vpip": vpip(hands), "pfr": pfr(hands), "three_bet": three_bet(hands),
        "fold_to_3bet": fold_to_3bet(hands), "four_bet": four_bet(hands),
        "cbet": cbet(hands), "fold_to_cbet": fold_to_cbet(hands),
        "wtsd": wtsd(hands), "won_at_showdown": won_at_showdown(hands),
        "wwsf": wwsf(hands), "aggression_factor": aggression_factor(hands),
        "aggression_frequency": aggression_frequency(hands),
        "bb_per_100": bb_per_100(hands), "roi": roi(hands),
    })
    if tournament_results:
        b.metrics["itm"] = itm(tournament_results)
    return b
