"""Deviation Agent (策略随机偏移): permanently separates base vs deviated.

Per spec §I the raw Judge recommendation (base) is *never* overwritten: this
agent either passes the base action through unchanged or picks a DIFFERENT
legal action and marks `deviated=True`.  The RNG is seeded from config
(`deviation.seed`), null meaning true system entropy.

Input : {state: dict, judge: dict|None}
Output: DeviationOutput (below)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext


@dataclass
class DeviationOutput:
    deviated: bool = False
    deviation_probability: float = 0.0
    original_action: str = "fold"
    original_size: float | None = None
    final_action: str = "fold"
    final_size: float | None = None
    reason: str = ""
    experiment_mode: bool = False
    confidence: float = 1.0
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


ACTIONS: list[str] = ["fold", "check", "call", "bet", "raise", "allin"]


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


class DeviationAgent(BaseAgent):
    name: ClassVar[str] = "deviation"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "策略随机偏移 Agent"
    layer: ClassVar[str] = "decision"
    depends: ClassVar[list[str]] = ["state", "judge"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = DeviationOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        # Always runs: with a missing judge or disabled config it passes through.
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> DeviationOutput:
        state = inp.get("state") or {}
        judge = inp.get("judge")
        out = DeviationOutput()
        warnings: list[dict] = []

        dev_cfg = ctx.config.deviation_config()
        enabled = bool(dev_cfg.get("enabled", True))
        probability = _num(dev_cfg.get("probability"))
        p = probability if probability is not None else 0.15
        min_p = _num(dev_cfg.get("min_probability"))
        max_p = _num(dev_cfg.get("max_probability"))
        min_p = min_p if min_p is not None else 0.0
        max_p = max_p if max_p is not None else 0.20
        seed = dev_cfg.get("seed")
        experiment_mode = bool(dev_cfg.get("experiment_mode", False))

        # ---- original action/size from judge (never overwritten) ----
        original_action = (judge or {}).get("action") if isinstance(judge, dict) else None
        if original_action not in ACTIONS:
            original_action = "fold"
        original_size = None
        if isinstance(judge, dict):
            original_size = _num(judge.get("recommended_size"))
            if original_size is None:
                original_size = _num(judge.get("recommended_size_bb"))
        out.original_action = original_action
        out.original_size = original_size
        out.experiment_mode = experiment_mode
        out.deviation_probability = round(_clip(p, 0.0, 1.0), 4)

        # ---- passthrough: disabled or judge missing ----
        if judge is None or not enabled:
            out.deviated = False
            out.final_action = original_action
            out.final_size = original_size
            out.reason = "deviation disabled or judge missing"
            out.evidence = [{"kind": "deviation", "text": f"original_action={original_action}, probability={p}",
                             "source": "deviation", "weight": 1.0}]
            out.warnings = warnings
            return out

        # ---- clamp probability unless experiment_mode ----
        if not experiment_mode and (p < min_p or p > max_p):
            warnings.append({
                "code": "DEVIATION_PROB_CLIPPED",
                "message": f"deviation probability {p} outside [{min_p},{max_p}], clipped to {_clip(p, min_p, max_p)}",
            })
            p = _clip(p, min_p, max_p)
        elif experiment_mode:
            warnings.append({
                "code": "EXPERIMENT_MODE",
                "message": f"实验模式开启：允许任意 deviation probability（当前 p={p}）",
            })
        out.deviation_probability = round(_clip(p, 0.0, 1.0), 4)

        # ---- seeded RNG ----
        rng = random.Random(seed) if seed is not None else random.Random()
        roll = rng.random()

        if roll >= p:
            out.deviated = False
            out.final_action = original_action
            out.final_size = original_size
            out.reason = "no deviation"
        else:
            available = [a for a in (state.get("available_actions") or ACTIONS) if a in ACTIONS]
            pool = [a for a in available if a != original_action]
            if not pool:
                out.deviated = False
                out.final_action = original_action
                out.final_size = original_size
                out.reason = "deviation triggered but no legal alternative action"
                warnings.append({"code": "NO_ALTERNATIVE_ACTION",
                                 "message": f"no legal action other than {original_action}"})
            else:
                # prefer the action most distant from the original in aggression
                # ordering (fold <-> raise/allin), tie-broken by the seeded RNG
                idx = {a: i for i, a in enumerate(ACTIONS)}
                dists = {a: abs(idx[a] - idx[original_action]) for a in pool}
                maxd = max(dists.values())
                candidates = sorted(a for a in pool if dists[a] == maxd)
                final = rng.choice(candidates)

                out.deviated = True
                out.final_action = final
                if final in ("bet", "raise"):
                    if original_size is not None and original_size > 0:
                        out.final_size = original_size
                    else:
                        bb = float(state.get("bb") or 0.0)
                        if bb <= 0:
                            bb = 1.0
                        out.final_size = round(0.5 * float(state.get("pot") or 0.0) / bb, 3)
                else:
                    out.final_size = None
                out.reason = f"deviation triggered (roll={roll:.4f} < p={p:.4f}): {original_action} -> {final}"
                warnings.append({
                    "code": "DEVIATION_APPLIED",
                    "message": (f"当前建议发生策略偏离（deviation={p * 100:.0f}%）："
                                f"原建议 {original_action} → 最终 {final}；"
                                f"如未偏离应执行 {original_action}"),
                })

        out.evidence = [{
            "kind": "deviation",
            "text": f"original_action={original_action}, probability={p}, deviated={out.deviated}",
            "source": "deviation",
            "weight": 1.0,
        }]
        out.warnings = warnings
        return out
