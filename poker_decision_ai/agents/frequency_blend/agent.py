"""P30-9-BR：后置补全 agent——把**局面特异的范围结构弃牌率**混入 villain_frequency_prior。

为什么需要它（已证）：`frequency_prior` 跑在 `range` 之前 ⇒ 它拿不到对手范围；
而冻结 judge 的 fold-equity 通道读 `villains[pid].frequencies.fold.rate`
（当前几乎完全来自 n≈85k 的**全级别通用**群体统计，`w_stat≈0.9988`）。

本 agent 位于 `range` 之后、`judge` 之前：
  * 输入 `{state, range, frequency_prior}`；
  * 输出与 FrequencyPriorOutput 同构（judge 无需改动）；
  * **默认零改动**：`frequency_prior.range_blend_weight = 0` ⇒ 原样透传；
  * 权重 > 0 ⇒ `fold = (1−w)·群体 + w·范围结构`，来源标 `population×street+range`。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext

VERSION = "p30-9-br-frequency-blend-v1"


@dataclass
class FrequencyBlendOutput:
    available: bool = False
    selection: dict = field(default_factory=dict)
    villains: dict = field(default_factory=dict)
    confidence: float = 0.0
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    blend: dict = field(default_factory=dict)


def _weight(ctx: Any) -> float:
    import os
    try:
        cfg = ctx.config.frequency_prior_config() or {}
    except Exception:                                          # noqa: BLE001
        cfg = {}
    try:
        return float(os.environ.get("FP_RANGE_BLEND")
                      or cfg.get("range_blend_weight", 0.0) or 0.0)
    except Exception:                                          # noqa: BLE001
        return 0.0


def _range_fold(state: dict, range_in: dict, pid: str) -> tuple[float | None, dict]:
    from poker_decision_ai.domain.poker.fold_equity import FoldEstimator
    dist = ((range_in.get("villains") or {}).get(pid) or {}).get("distribution")
    if not dist:
        return None, {"reason": "no villain distribution from range agent"}
    players = state.get("players") or []
    hero = next((p for p in players if isinstance(p, dict) and p.get("is_hero")), {})
    bb = float(state.get("bb") or 0.0) or 1.0
    pot_bb = float(state.get("pot") or 0.0) / bb
    to_call_bb = float(state.get("to_call") or 0.0) / bb
    est = FoldEstimator(pot_bb=pot_bb,
                         facing="raise" if to_call_bb > 0 else "bet",
                         street=str(state.get("street") or "flop"),
                         villain_dist=dist,
                         hero_cards=[str(c) for c in (hero.get("hole_cards") or [])],
                         board=[str(c) for c in (state.get("board") or [])])
    res = est.for_size(to_call_bb if to_call_bb > 0 else max(0.5 * pot_bb, 0.5))
    if not getattr(res, "available", False) or res.p_fold is None:
        return None, {"reason": getattr(res, "unavailable_reason", "unavailable")}
    return float(res.p_fold), {"mode": getattr(res, "mode", "range"),
                                "provenance": getattr(res, "provenance", {})}


class FrequencyBlendAgent(BaseAgent):
    name: ClassVar[str] = "frequency_blend"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "频率先验-范围结构补全 Agent"
    layer: ClassVar[str] = "human"
    depends: ClassVar[list[str]] = ["state", "range"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = FrequencyBlendOutput

    async def applicable(self, raw_input: dict,
                          ctx: AgentContext) -> tuple[bool, str]:
        # 始终运行：即使不混合也要**原样透传**（保证 judge 不丢输入）
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> FrequencyBlendOutput:
        prior = inp.get("frequency_prior")
        prior = dict(prior) if isinstance(prior, dict) else {}
        state = inp.get("state") or {}
        range_in = inp.get("range") or {}
        w = _weight(ctx)
        out = FrequencyBlendOutput(
            available=bool(prior.get("available")),
            selection=dict(prior.get("selection") or {}),
            villains={k: dict(v) for k, v in (prior.get("villains") or {}).items()
                       if isinstance(v, dict)},
            confidence=float(prior.get("confidence") or 0.0),
            evidence=list(prior.get("evidence") or []),
            # 上游 warnings 已在 frequency_prior 的 envelope 里（不重复透传，
            # 否则本 agent 会被判 degraded，污染 F5 组件降级清单）
            warnings=[],
            blend={"weight": w, "version": VERSION})
        if w <= 0 or not out.villains:
            return out                                  # 默认：零改动透传
        changed = 0
        for pid, entry in out.villains.items():
            freq = entry.get("frequencies")
            f_stat = None
            if isinstance(freq, dict):
                try:
                    f_stat = float((freq.get("fold") or {}).get("rate"))
                except (TypeError, ValueError):
                    f_stat = None
            f_range, prov = _range_fold(state, range_in, str(pid))
            if f_range is None:
                out.warnings.append({"code": "FOLD_RANGE_UNAVAILABLE",
                                      "message": f"villain {pid}: {prov.get('reason')}"})
                continue
            if f_stat is None:
                f_final, src = f_range, "range_only"
            else:
                f_final, src = (1.0 - w) * f_stat + w * f_range, "population×street+range"
            f_final = min(max(f_final, 0.02), 0.98)
            entry["frequencies"] = {
                "fold": {"rate": round(f_final, 4)},
                "call": {"rate": round((1.0 - f_final) * 0.7, 4)},
                "raise": {"rate": round((1.0 - f_final) * 0.3, 4)},
                "bet": {"rate": round(1.0 - f_final, 4)},
            }
            entry["p_fold"] = round(f_final, 4)
            entry["p_continue"] = round(1.0 - f_final, 4)
            entry["source"] = src
            entry["range_fold_provenance"] = {
                "stat_fold": (round(f_stat, 4) if f_stat is not None else None),
                "range_fold": round(f_range, 4), "weight": w,
                "mode": prov.get("mode"), "provenance": prov.get("provenance"),
                "note": "群体统计与范围结构估计的混合（P30-9-BR）"}
            changed += 1
        if changed:
            out.available = True
            out.confidence = max(out.confidence, 0.5)
            out.evidence.append({
                "kind": "frequency_blend",
                "text": (f"范围结构补全：{changed} 个对手，权重 w={w}"
                          "（fold 来自群体统计与局面特异范围结构的混合）"),
                "weight": 1.0, "source": "frequency_blend"})
            out.blend["changed"] = changed
        return out
