"""Confidence Agent (动态权重，Judge 的信任分配器)。

Input : {state: dict, opponent: dict|None, population: dict|None, statistics: dict|None}
Output: ConfidenceOutput {players: {pid: {...}}, hero_weights, confidence, evidence, warnings}

动态权重核心公式（写注释）：
  hands_observed        = max(opponent.sample_size, statistics 里该玩家最大样本量)
  individual_confidence = sigmoid((hands - c) / k),  c=150, k=250
                          （15 手 ≈ 0.37；5000 手 ≈ 1.0）
  data_quality          = min(1, 有效指标数/10) * 新鲜度
                          新鲜度 = last_seen 距今天数按 90 天半衰期指数衰减；未知记 0.8
  recency_score         = recent samples / total samples；无数据 0.5
  individual_w          = clip(individual_confidence * data_quality * recency_score, 0, 0.8)
  type_w                = 0.25 * (1 - individual_w)          # 类型层基于 feature_vector 稳定性
  population_w          = 1 - individual_w - type_w           # 三权重归一（和 = 1）
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext

_C = 150.0        # sigmoid 中心（手数）
_K = 250.0        # sigmoid 陡度
_HALF_LIFE_DAYS = 90.0


@dataclass
class ConfidenceOutput:
    players: dict = field(default_factory=dict)   # pid -> {hands_observed, individual_confidence, data_quality, recency_score, weights}
    hero_weights: dict = field(default_factory=dict)  # {"population","type","individual"}
    confidence: float = 0.5
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class ConfidenceAgent(BaseAgent):
    name: ClassVar[str] = "confidence"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "动态权重 Agent"
    layer: ClassVar[str] = "analysis"
    depends: ClassVar[list[str]] = ["state", "opponent", "statistics"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = ConfidenceOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        state = raw_input.get("state") or {}
        if not state.get("hero_id") and not state.get("players"):
            return False, "state has no players to weight"
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> ConfidenceOutput:
        state = inp.get("state") or {}
        opponent = inp.get("opponent") or {}
        statistics = inp.get("statistics") or {}
        warnings: list[dict] = []

        hero_id = str(state.get("hero_id") or "")
        players = state.get("players") or []
        targets: list[str] = [hero_id] if hero_id else []
        for p in players:
            pid = str(p.get("player_id"))
            status = p.get("status") if isinstance(p, dict) else getattr(p, "status", "active")
            if status in ("active", "allin") and pid != hero_id:
                targets.append(pid)
        targets = list(dict.fromkeys(targets))

        players_out: dict[str, dict] = {}
        for pid in targets:
            players_out[pid] = self._weight_for(pid, opponent, statistics)

        hero_weights = players_out[hero_id]["weights"] if hero_id and hero_id in players_out else _weights(0, 0.0, 0.0, 0.5)
        confidence = players_out[hero_id]["individual_confidence"] if hero_id and hero_id in players_out else 0.5

        return ConfidenceOutput(
            players=players_out, hero_weights=hero_weights,
            confidence=round(float(confidence), 4),
            evidence=[{"kind": "confidence", "source": "opponent|statistics",
                       "text": f"为 {len(players_out)} 名玩家计算动态权重", "weight": 1.0}],
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    def _weight_for(self, pid: str, opponent: dict, statistics: dict) -> dict:
        hands = _hands_observed(pid, opponent, statistics)
        ind_conf = _individual_confidence(hands)
        quality = _data_quality(pid, opponent)
        recency = _recency_score(pid, opponent)
        weights = _weights(hands, ind_conf, quality, recency)
        return {
            "hands_observed": int(hands),
            "individual_confidence": round(ind_conf, 4),
            "data_quality": round(quality, 4),
            "recency_score": round(recency, 4),
            "weights": weights,
        }


# --------------------------------------------------------------------------- #
# 模块级 helpers
# --------------------------------------------------------------------------- #
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _individual_confidence(hands: float, c: float = _C, k: float = _K) -> float:
    """sigmoid((hands - c)/k)。15 手 ≈ 0.37；5000 手 ≈ 1.0。"""
    return _sigmoid((hands - c) / k)


def _hands_observed(pid: str, opponent: dict, statistics: dict) -> float:
    """取 opponent.sample_size 与 statistics 里该玩家最大样本量中的较大者。"""
    best = 0.0
    opp_players = opponent.get("players", {}) if isinstance(opponent, dict) else {}
    opp_p = opp_players.get(pid) if isinstance(opp_players, dict) else None
    if isinstance(opp_p, dict):
        best = max(best, float(opp_p.get("sample_size") or 0))

    if isinstance(statistics, dict):
        sources = []
        if pid and statistics.get("hero_stats") and isinstance(statistics["hero_stats"], dict):
            sources.append(statistics["hero_stats"])
        villains = statistics.get("villains") if isinstance(statistics.get("villains"), dict) else {}
        if isinstance(villains, dict) and villains.get(pid):
            sources.append(villains[pid])
        for src in sources:
            if isinstance(src, dict):
                for m in src.values():
                    if isinstance(m, dict) and m.get("samples"):
                        best = max(best, float(m["samples"]))
    return best


def _data_quality(pid: str, opponent: dict) -> float:
    """= min(1, 有效指标数/10) * 新鲜度（90 天半衰期，未知 0.8）。"""
    valid = 0
    opp_players = opponent.get("players", {}) if isinstance(opponent, dict) else {}
    opp_p = opp_players.get(pid) if isinstance(opp_players, dict) else None
    behavior = opp_p.get("behavior_statistics", {}) if isinstance(opp_p, dict) else {}
    if isinstance(behavior, dict):
        for m in behavior.values():
            if isinstance(m, dict) and int(m.get("samples") or 0) > 0:
                valid += 1
    metric_quality = min(1.0, valid / 10.0)

    freshness = _freshness(opp_p)
    return metric_quality * freshness


def _freshness(opp_p: dict | None) -> float:
    """last_seen 距今天数的指数衰减（90 天半衰期）；未知记 0.8。"""
    if not isinstance(opp_p, dict):
        return 0.8
    last_seen = opp_p.get("last_seen")
    if not last_seen:
        return 0.8
    try:
        dt = datetime.fromisoformat(str(last_seen).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days = max(0.0, (now - dt).total_seconds() / 86400.0)
    except (ValueError, TypeError):
        return 0.8
    return 0.5 ** (days / _HALF_LIFE_DAYS)


def _recency_score(pid: str, opponent: dict) -> float:
    """recent 样本量映射：无 recent 数据 0.5 中性；>=300 手近期样本 = 1.0。

    采用 0.5 + 0.5*min(1, recent/300) 而不是 recent/total：前者对大样本
    历史（如 5000 手、最近 300 手）给 1.0，符合"大量可靠数据 → individual 0.75+"
    的契约目标；纯比值 recent/total 会把大样本错误压到 ~0。
    """
    opp_players = opponent.get("players", {}) if isinstance(opponent, dict) else {}
    opp_p = opp_players.get(pid) if isinstance(opp_players, dict) else None
    if not isinstance(opp_p, dict):
        return 0.5
    recent = opp_p.get("recent_behavior") if isinstance(opp_p.get("recent_behavior"), dict) else {}
    recent_samples = 0
    if isinstance(recent, dict):
        for m in recent.values():
            if isinstance(m, dict):
                recent_samples = max(recent_samples, int(m.get("samples") or 0))
    if recent_samples <= 0:
        return 0.5            # 无近期数据 → 中性，不惩罚
    return min(1.0, 0.5 + 0.5 * recent_samples / 300.0)


def _weights(hands: float, ind_conf: float, quality: float, recency: float) -> dict:
    """三权重：individual / type / population，和为 1。"""
    individual_w = _clip01(ind_conf * quality * recency, hi=0.8)
    type_w = 0.25 * (1.0 - individual_w)
    population_w = 1.0 - individual_w - type_w
    # 归一化到三位小数（population 由 1 - ind - type 反推，保证和 = 1）
    individual_w = round(individual_w, 3)
    type_w = round(type_w, 3)
    population_w = round(1.0 - individual_w - type_w, 3)
    return {"individual": individual_w, "type": type_w, "population": population_w}


def _clip01(x: float, hi: float = 1.0) -> float:
    return max(0.0, min(hi, x))
