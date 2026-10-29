"""Experiment Agent (实验): compare strategy arms on decision-quality outcomes.

Reads historical decisions from `ctx.repos.decisions`, groups them by arm and
compares mean decision-quality score per arm using a simple two-sample z test
(no new dependencies: the standard-normal CDF uses the stdlib `math.erf`).

Input : {request: dict|None, research: dict|None, evaluation: dict|None}
Output: ExperimentOutput (below)
"""
from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext


@dataclass
class ExperimentOutput:
    experiments: list = field(default_factory=list)   # [{id,name,arms,results,winner,significance,status,note}]
    confidence: float = 0.5
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


DEFAULT_ARMS = ["gto_only", "gto_population", "gto_player", "gto_player_icm"]
MIN_SAMPLES = 30      # per-arm sample threshold before a winner may be declared
Z_THRESHOLD = 1.96    # two-tailed 95% significance


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erf: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def _extract_arm(decision: dict) -> str | None:
    """Arm of a stored decision: request.arm / meta.arm / top-level arm."""
    request = decision.get("request")
    if isinstance(request, dict) and request.get("arm"):
        return str(request["arm"])
    meta = decision.get("meta")
    if isinstance(meta, dict) and meta.get("arm"):
        return str(meta["arm"])
    if decision.get("arm"):
        return str(decision["arm"])
    return None


def _extract_score(decision: dict) -> float | None:
    """Decision quality score: decision.evaluation.decision_quality.score."""
    eval_ = decision.get("evaluation")
    if not isinstance(eval_, dict):
        return None
    dq = eval_.get("decision_quality")
    if not isinstance(dq, dict):
        return None
    score = dq.get("score")
    if isinstance(score, bool):
        return None
    if isinstance(score, (int, float)):
        return float(score)
    return None


class ExperimentAgent(BaseAgent):
    name: ClassVar[str] = "experiment"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "实验 Agent"
    layer: ClassVar[str] = "experiment"
    depends: ClassVar[list[str]] = ["evaluation"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = ExperimentOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        # Always applicable: it reads the decision repository, not just input.
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> ExperimentOutput:
        out = ExperimentOutput()
        warnings: list[dict] = []

        decisions = ctx.repos.decisions.list()

        # ---- group scores by arm ----
        scores_by_arm: dict[str, list[float]] = {}
        for d in decisions:
            arm = _extract_arm(d)
            score = _extract_score(d)
            if arm is not None and score is not None:
                scores_by_arm.setdefault(arm, []).append(score)

        if not scores_by_arm:
            warnings.append({"code": "NO_EXPERIMENT_DATA",
                             "message": "no decisions with arm + decision-quality score available"})
            out.confidence = 0.0
            out.evidence = [{"kind": "experiment", "text": "no arm data", "source": "experiment", "weight": 1.0}]
            out.warnings = warnings
            return out

        # ---- per-arm summary stats ----
        results: dict[str, dict] = {}
        for arm, vals in scores_by_arm.items():
            n = len(vals)
            mean = float(statistics.fmean(vals))
            std = float(statistics.stdev(vals)) if n >= 2 else 0.0
            results[arm] = {"n": n, "mean": round(mean, 4), "std": round(std, 4)}

        # ---- comparison: require >=2 arms with n >= MIN_SAMPLES ----
        qualified = [arm for arm, r in results.items() if r["n"] >= MIN_SAMPLES]
        winner: str | None = None
        significance = 0.0
        note = ""
        status = "running"

        if len(qualified) < 2:
            status = "running"
            note = (f"样本不足：需要至少 2 个 arm 各 ≥{MIN_SAMPLES} 条决策，"
                    f"当前各 arm 样本数={results}")
            significance = 0.0
        else:
            # compare the two arms with the largest means (z test on means)
            ordered = sorted(qualified, key=lambda a: results[a]["mean"], reverse=True)
            a, b = ordered[0], ordered[1]
            ma, mb = results[a]["mean"], results[b]["mean"]
            sa, sb = results[a]["std"], results[b]["std"]
            na, nb = results[a]["n"], results[b]["n"]
            # standard error of the difference of two means
            se = math.sqrt((sa ** 2 / na) + (sb ** 2 / nb))
            status = "concluded"
            if se <= 0:
                # degenerate: zero within-arm variance -> deterministically
                # decided by the mean (all samples of each arm are identical)
                if ma > mb:
                    winner = a
                    significance = 1.0
                    note = f"{a} 确定性优于 {b}（两组内零方差，均值 {ma:.3f} > {mb:.3f}）。"
                else:
                    winner = None
                    significance = 0.0
                    note = "无显著差异（方差为 0，两组均值相同）。"
            else:
                z = abs(ma - mb) / se
                # two-tailed significance = 2*Phi(z) - 1
                significance = round(_clip(2 * _norm_cdf(z) - 1), 4)
                if z > Z_THRESHOLD:
                    winner = a
                    note = (f"{a} 显著优于 {b}（z={z:.2f} > {Z_THRESHOLD}，"
                            f"均值 {ma:.3f} vs {mb:.3f}）。")
                else:
                    winner = None
                    note = f"无显著差异（z={z:.2f} ≤ {Z_THRESHOLD}），无法宣布 winner。"

        exp_id = f"exp-{int(time.time() * 1000)}"
        experiment = {
            "id": exp_id,
            "name": "strategy_arm_comparison",
            "arms": list(DEFAULT_ARMS),
            "results": results,
            "winner": winner,
            "significance": significance,
            "status": status,
            "note": note,
        }
        out.experiments = [experiment]

        # ---- persist into research knowledge base ----
        try:
            ctx.repos.research.add("experiment", dict(experiment))
        except Exception as e:
            warnings.append({"code": "RESEARCH_SAVE_FAILED", "message": str(e)})

        # ---- confidence grows with the smallest qualified arm sample ----
        if qualified:
            min_n = min(results[a]["n"] for a in qualified)
            out.confidence = round(_clip(min_n / (MIN_SAMPLES * 2)), 3)
        else:
            max_n = max((r["n"] for r in results.values()), default=0)
            out.confidence = round(_clip(max_n / (MIN_SAMPLES * 2)), 3)

        out.evidence = [{
            "kind": "experiment",
            "text": f"arms={sorted(scores_by_arm)}, winner={winner}, significance={significance}",
            "source": "experiment",
            "weight": 1.0,
        }]
        out.warnings = warnings
        return out
