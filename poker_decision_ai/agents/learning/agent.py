"""Learning Agent (学习): profile-update engine with over-fitting protection.

Core anti-over-fitting rules (each documented inline):
  1. New sample count < min_sample_size  -> reject("样本不足").
  2. Wilson interval of the new rate overlaps the archived rate's interval
     -> reject("置信区间重叠，无显著差异"); only non-overlap -> update.
  3. Bad-beat protection: a single hand that lost is NEVER used to change
     strategy (only recorded as a note).
  4. Time decay: merged value = bayes_shrink(new, new_n, old, strength=old_n *
     decay_factor) with decay_factor = 0.5 ** (days / half_life).
  5. request.skip_learning -> every candidate decision is "reject".
Updates are persisted through ctx.repos.players.upsert /
ctx.repos.populations.upsert_segment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.core.schemas import utc_now_iso
from poker_decision_ai.domain.poker import bayes_shrink, wilson_interval


def _decay_factor(last_seen: str | None, now_iso: str, half_life_days: float) -> float:
    """0.5 ** (days_elapsed / half_life) -> how much of the old sample counts.

    Older observations count for less (exponential time decay).  If the
    timestamp is missing or unparseable we conservatively keep full weight
    (1.0) rather than discarding data silently.
    """
    if not last_seen:
        return 1.0
    try:
        t = datetime.fromisoformat(str(last_seen).replace("Z", "+00:00"))
        n = datetime.fromisoformat(str(now_iso).replace("Z", "+00:00"))
        days = max(0.0, (n - t).total_seconds() / 86400.0)
        return 0.5 ** (days / max(half_life_days, 1.0))
    except Exception:
        return 1.0


@dataclass
class LearningOutput:
    updates: list = field(default_factory=list)     # [{target,id,metric,old_value,new_value,samples,significance,decision,reason}]
    rejected: list = field(default_factory=list)    # [{target,metric,reason}]
    bad_beat_protection: dict = field(default_factory=lambda: {"triggered": False, "note": ""})
    confidence: float = 1.0
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class LearningAgent(BaseAgent):
    name: ClassVar[str] = "learning"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "学习 Agent"
    layer: ClassVar[str] = "learning"
    depends: ClassVar[list[str]] = ["state", "judge", "deviation", "statistics",
                                    "opponent", "population", "confidence", "request"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = LearningOutput

    async def handle(self, inp: dict, ctx: AgentContext) -> LearningOutput:
        cfg = ctx.config.learning_config()
        min_sample = int(cfg.get("min_sample_size", 50))
        z = float(cfg.get("significance_z", 1.96))
        half_life = float(cfg.get("decay_half_life_days", 90))
        # population_prior_strength 预留给“向群体均值收缩”的路径（此处未直接使用，
        # 因学习层更新是对“新旧档案”做差异检验，而非向群体先验收缩）。
        _prior_strength = int(cfg.get("population_prior_strength", 200))

        state = inp.get("state") or {}
        request = inp.get("request") or {}
        statistics = inp.get("statistics") or {}
        opponent = inp.get("opponent") or {}

        out = LearningOutput()
        skip = bool(request.get("skip_learning"))
        hero_id = str(state.get("hero_id") or "hero")
        now_iso = utc_now_iso()

        # ---- bad-beat protection ----
        # 本次单手若亏损，绝不因“结果”更新策略。注：equity 不在本 Agent 输入中，
        # 这里以 request.outcome 的亏损作为触发条件（若上游提供 equity，则由其前置判断）。
        bb_protection = {"triggered": False, "note": ""}
        outcome = request.get("outcome")
        if isinstance(outcome, dict):
            result = outcome.get("result")
            won = outcome.get("won")
            if result in ("lost", "LOST") or won is False:
                bb_protection = {
                    "triggered": True,
                    "note": "本手亏损：单手结果不用于更新策略（防过拟合，避免把坏运气写进档案）",
                }
        out.bad_beat_protection = bb_protection

        # ---- 更新候选：hero 与 villains 的新统计 (statistics) vs 旧档案 (players repo) ----
        candidates: list[tuple[str, str, str, float, int]] = []  # (pid, metric, new_val, new_n)
        hero_stats = statistics.get("hero_stats") if isinstance(statistics, dict) else {}
        if isinstance(hero_stats, dict):
            for metric, m in hero_stats.items():
                if not isinstance(m, dict):
                    continue
                try:
                    new_n = int(m.get("samples") or 0)
                    new_val = float(m.get("value") or 0.0)
                except (TypeError, ValueError):
                    continue
                if new_n > 0:
                    candidates.append((hero_id, metric, new_val, new_n))
        villains = statistics.get("villains") if isinstance(statistics, dict) else {}
        if isinstance(villains, dict):
            for pid, stats in villains.items():
                if not isinstance(stats, dict):
                    continue
                for metric, m in stats.items():
                    if not isinstance(m, dict):
                        continue
                    try:
                        new_n = int(m.get("samples") or 0)
                        new_val = float(m.get("value") or 0.0)
                    except (TypeError, ValueError):
                        continue
                    if new_n > 0:
                        candidates.append((str(pid), metric, new_val, new_n))

        for pid, metric, new_val, new_n in candidates:
            # 规则 5：跳过学习 -> 全部 reject
            if skip:
                out.updates.append({
                    "target": "player", "id": pid, "metric": metric,
                    "old_value": 0.0, "new_value": new_val, "samples": new_n,
                    "significance": 0.0, "decision": "reject",
                    "reason": "skip_learning=True",
                })
                out.rejected.append({"target": pid, "metric": metric,
                                     "reason": "skip_learning=True"})
                continue

            old = ctx.repos.players.get(pid) or {}
            old_bs = old.get("behavior_statistics") or {}
            old_metric = old_bs.get(metric) if isinstance(old_bs, dict) else None
            old_val = 0.0
            old_n = 0
            if isinstance(old_metric, dict):
                try:
                    old_val = float(old_metric.get("value") or 0.0)
                    old_n = int(old_metric.get("samples") or 0)
                except (TypeError, ValueError):
                    old_val, old_n = 0.0, 0

            # 规则 1：样本不足
            if new_n < min_sample:
                out.updates.append({
                    "target": "player", "id": pid, "metric": metric,
                    "old_value": old_val, "new_value": new_val, "samples": new_n,
                    "significance": 0.0, "decision": "reject",
                    "reason": f"样本不足 (n={new_n} < {min_sample})",
                })
                out.rejected.append({"target": pid, "metric": metric,
                                     "reason": f"样本不足 (n={new_n} < {min_sample})"})
                continue

            if old_n <= 0:
                out.updates.append({
                    "target": "player", "id": pid, "metric": metric,
                    "old_value": 0.0, "new_value": new_val, "samples": new_n,
                    "significance": 0.0, "decision": "reject",
                    "reason": "无旧档案可比，拒绝在无基准时更新",
                })
                out.rejected.append({"target": pid, "metric": metric,
                                     "reason": "无旧档案可比"})
                continue

            # 规则 2：Wilson 区间重叠检验
            lo_old, hi_old = wilson_interval(int(round(old_val * old_n)), old_n, z)
            lo_new, hi_new = wilson_interval(int(round(new_val * new_n)), new_n, z)
            overlap = max(0.0, min(hi_old, hi_new) - max(lo_old, lo_new))
            width = (hi_new - lo_new) or 1e-9
            # significance = 1 - 重叠占新区间宽度的比例（无重叠 -> 1.0）
            significance = max(0.0, min(1.0, 1.0 - overlap / width))

            if overlap > 1e-9:
                out.updates.append({
                    "target": "player", "id": pid, "metric": metric,
                    "old_value": old_val, "new_value": new_val, "samples": new_n,
                    "significance": round(significance, 4), "decision": "reject",
                    "reason": "置信区间重叠，无显著差异",
                })
                out.rejected.append({"target": pid, "metric": metric,
                                     "reason": "置信区间重叠，无显著差异"})
                continue

            # ---- 执行更新 ----
            # 规则 4：时间衰减合并。
            # strength = 旧样本 * 衰减因子（衰减因子随时间减半）。
            decay_factor = _decay_factor(old.get("last_seen"), now_iso, half_life)
            strength = max(1, int(round(old_n * decay_factor)))
            merged = bayes_shrink(new_val, new_n, old_val, strength)

            merged_bs = dict(old_bs) if isinstance(old_bs, dict) else {}
            merged_bs[metric] = {"value": round(merged, 4), "samples": old_n + new_n}

            fv = {}
            opp_players = opponent.get("players") if isinstance(opponent, dict) else {}
            if isinstance(opp_players, dict) and isinstance(opp_players.get(pid), dict):
                fv = opp_players[pid].get("feature_vector") or {}
            if not fv:
                fv = old.get("feature_vector") or {}

            ctx.repos.players.upsert({
                "player_id": pid,
                "feature_vector": fv,
                "behavior_statistics": merged_bs,
                "sample_size": int(old.get("sample_size") or 0) + new_n,
                "last_seen": now_iso,
            })

            # 群体更新（尽力而为）：把显著变化的指标合并进 primary_segment。
            seg_key = population_seg_key(inp)
            if seg_key:
                ctx.repos.populations.upsert_segment(
                    seg_key, {metric: {"value": round(merged, 4), "samples": old_n + new_n}})

            out.updates.append({
                "target": "player", "id": pid, "metric": metric,
                "old_value": old_val, "new_value": round(merged, 4), "samples": new_n,
                "significance": round(significance, 4), "decision": "update",
                "reason": "新旧置信区间不重叠，存在显著变化",
            })

        out.evidence = [{
            "kind": "learning",
            "text": f"updates={sum(1 for u in out.updates if u.get('decision') == 'update')}, "
                    f"rejected={len(out.rejected)}, skip_learning={skip}",
            "source": "learning",
            "weight": 1.0,
        }]
        if skip:
            out.warnings.append({"code": "SKIP_LEARNING",
                                 "message": "request.skip_learning=True，所有更新被拒绝"})
        return out


def population_seg_key(inp: dict) -> str | None:
    """Best-effort primary segment key for the population upsert."""
    population = inp.get("population")
    if isinstance(population, dict):
        key = population.get("primary_segment")
        if key:
            return str(key)
        segs = population.get("segments")
        if isinstance(segs, dict) and segs:
            return str(next(iter(segs)))
    return None
