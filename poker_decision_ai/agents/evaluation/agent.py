"""Evaluation Agent (系统评估): decision quality != short-term outcome.

Core idea (spec §4 evaluation): a good decision can lose and a bad decision
can win.  The agent therefore scores *decision quality* independently of the
hand result, and only afterwards compares the two.

Input : {state: dict, judge: dict|None, deviation: dict|None, request: dict|None}
Output: EvaluationOutput (below)

decision_quality.score is a deterministic weighted blend (formula documented
in the comments below); no LLM is used for any number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext


@dataclass
class EvaluationOutput:
    decision_quality: dict = field(default_factory=dict)   # {"score": 0..1, "factors": [str]}
    prediction_accuracy: dict | None = None                # {"bluff_correct": float|None, "samples": int}
    calibration: dict | None = None                        # {"brier_score": float|None, "samples": int}
    outcome: dict | None = None                            # {"result","net_bb","won"}
    decision_vs_outcome: dict = field(default_factory=dict)  # {"decision_good","outcome_good","note"}
    confidence: float = 0.5
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


class EvaluationAgent(BaseAgent):
    name: ClassVar[str] = "evaluation"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "系统评估 Agent"
    layer: ClassVar[str] = "experiment"
    depends: ClassVar[list[str]] = ["state", "judge", "deviation"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = EvaluationOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        judge = raw_input.get("judge")
        request = raw_input.get("request") or {}
        state = raw_input.get("state") or {}
        if judge is None and not request.get("outcome") and not state:
            return False, "nothing to evaluate (no judge / outcome / state)"
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> EvaluationOutput:
        judge = inp.get("judge")
        deviation = inp.get("deviation")
        request = inp.get("request") or {}
        state = inp.get("state") or {}
        out = EvaluationOutput()
        warnings: list[dict] = []

        # ------------------------------------------------------------------
        # 1) decision quality (deterministic synthesis).
        #    score = 0.25*confidence + 0.25*evidence_completeness
        #          + 0.25*source_completeness + 0.25*risk_discount
        #
        #    - confidence            : judge.confidence (fallback 0.5)
        #    - evidence_completeness : min(1, len(key_evidence)/5)  -> simplified
        #                              proxy for "equity gap" quality
        #    - source_completeness   : ratio of 'ok' in final_inputs_summary
        #    - risk_discount         : 0.5 neutral (no risk input in this agent's
        #                              contract); would be (1 - decision_risk)
        #                              if a numeric risk were available
        # ------------------------------------------------------------------
        judge_conf = 0.5
        key_evidence: list = []
        summary: dict = {}
        if isinstance(judge, dict):
            jc = judge.get("confidence")
            if isinstance(jc, (int, float)):
                judge_conf = float(jc)
            key_evidence = judge.get("key_evidence") or []
            summary = judge.get("final_inputs_summary") or {}

        comp_confidence = _clip(judge_conf)
        comp_evidence = _clip(len(key_evidence) / 5.0)
        ok = sum(1 for v in summary.values() if v == "ok")
        comp_sources = (ok / len(summary)) if summary else 0.0
        comp_sources = _clip(comp_sources)
        risk_discount = 0.5  # neutral: evaluation input carries no risk block

        score = round(_clip(
            0.25 * comp_confidence
            + 0.25 * comp_evidence
            + 0.25 * comp_sources
            + 0.25 * risk_discount
        ), 4)
        factors = [
            f"judge_confidence={judge_conf:.3f} (weight 0.25)",
            f"evidence_completeness={comp_evidence:.3f} ({len(key_evidence)}/5 key_evidence, weight 0.25)",
            f"source_completeness={comp_sources:.3f} ({ok}/{len(summary)} sources ok, weight 0.25)",
            f"risk_discount={risk_discount:.3f} (no risk input -> neutral 0.5, weight 0.25)",
        ]
        out.decision_quality = {"score": score, "factors": factors}

        # ------------------------------------------------------------------
        # 2) outcome (from request.outcome; optional)
        # ------------------------------------------------------------------
        outcome_in = request.get("outcome") if isinstance(request, dict) else None
        if isinstance(outcome_in, dict):
            result = str(outcome_in.get("result") or "unknown")
            if result not in ("won", "lost", "folded", "unknown"):
                result = "unknown"
            net_bb = outcome_in.get("net_bb")
            net_bb = float(net_bb) if isinstance(net_bb, (int, float)) else 0.0
            won = outcome_in.get("won")
            if won is None:
                won = True if result == "won" else (False if result == "lost" else None)
            else:
                won = bool(won)
            out.outcome = {"result": result, "net_bb": net_bb, "won": won}

        # ------------------------------------------------------------------
        # 3) calibration (brier score when an expected equity is present)
        #    brier = (expected_equity - actual)^2 ; actual = 1 if won else 0
        # ------------------------------------------------------------------
        expected = outcome_in.get("expected_equity") if isinstance(outcome_in, dict) else None
        if expected is not None and out.outcome is not None and out.outcome.get("won") is not None:
            actual = 1.0 if out.outcome["won"] else 0.0
            try:
                brier = (float(expected) - actual) ** 2
                out.calibration = {"brier_score": round(brier, 4), "samples": 1}
            except (TypeError, ValueError):
                out.calibration = None
        else:
            out.calibration = None

        # ------------------------------------------------------------------
        # 4) decision vs outcome (the separation the framework exists for)
        # ------------------------------------------------------------------
        decision_good = score > 0.6
        outcome_good = out.outcome.get("won") if out.outcome is not None else None
        if outcome_good is None:
            note = ("决策质量高（score>0.6），但本次无结果数据。" if decision_good
                    else "决策质量偏低，且无结果数据。")
        elif decision_good and outcome_good:
            note = "决策质量高且结果有利——策略与结果一致。"
        elif decision_good and not outcome_good:
            note = "决策质量高但短期结果不利——不修改策略（决策正确 ≠ 短期结果）。"
        elif not decision_good and outcome_good:
            note = "决策质量偏低但结果有利——警惕运气成分，应复盘决策过程而非结果。"
        else:
            note = "决策质量偏低且结果不利——应复盘改进。"
        out.decision_vs_outcome = {
            "decision_good": decision_good,
            "outcome_good": outcome_good,
            "note": note,
        }

        # ------------------------------------------------------------------
        # 5) persist decision (only once per request_id; merge evaluation in)
        # ------------------------------------------------------------------
        rid = str(ctx.request_id or request.get("request_id") or "unknown")
        try:
            existing = ctx.repos.decisions.get(rid)
            eval_dict = {
                "decision_quality": out.decision_quality,
                "prediction_accuracy": out.prediction_accuracy,
                "calibration": out.calibration,
                "outcome": out.outcome,
                "decision_vs_outcome": out.decision_vs_outcome,
            }
            if existing is None:
                ctx.repos.decisions.save({
                    "request_id": rid,
                    "base_decision": judge,
                    "deviated_decision": deviation,
                    "evaluation": eval_dict,
                })
            elif "evaluation" not in existing:
                merged = dict(existing)
                merged["evaluation"] = eval_dict
                ctx.repos.decisions.save(merged)
        except Exception as e:  # repository failure must never crash evaluation
            warnings.append({"code": "DECISION_SAVE_FAILED", "message": str(e)})

        # ------------------------------------------------------------------
        # 6) meta fields
        # ------------------------------------------------------------------
        out.confidence = round(_clip(0.5 + 0.5 * comp_sources), 3) if summary else 0.5
        out.evidence = [{
            "kind": "evaluation",
            "text": f"decision_quality score={score}, decision_good={decision_good}, outcome_good={outcome_good}",
            "source": "evaluation",
            "weight": 1.0,
        }]
        if not judge and not state and not out.outcome:
            warnings.append({"code": "NO_DECISION_INPUT", "message": "no judge/state/outcome; score is conservative"})
        out.warnings = warnings
        return out
