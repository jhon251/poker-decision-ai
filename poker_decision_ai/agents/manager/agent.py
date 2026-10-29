"""Manager Agent (总调度): the single orchestration entry point.

Responsibilities (spec §四-1):
  - decide the plan (from pipeline.yaml, by request kind)
  - call agents dynamically (applicable() checks), run parallel where possible
  - handle timeout / retry / degradation
  - aggregate results, hand the final decision to Judge -> Deviation
  - produce a structured summary envelope

The Manager performs NO poker analysis itself.  The heavy lifting lives in
core/orchestration (plan/engine/runner); this agent is its Agent-protocol
facade so the Manager itself is observable, loggable and testable like any
other agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.errors import OrchestrationError
from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.core.orchestration.plan import build_plan, plan_as_doc
from poker_decision_ai.core.orchestration.registry import register_all_agents
from poker_decision_ai.core.orchestration.runner import PipelineRunner


@dataclass
class ManagerOutput:
    kind: str
    status: str                                  # ok | degraded | error
    final_decision: dict | None = None
    base_decision: dict | None = None            # Judge 原始建议（永久区分，spec §I）
    deviated_decision: dict | None = None
    deviation_applied: bool = False
    agents_run: list = field(default_factory=list)
    agents_skipped: list = field(default_factory=list)
    agents_degraded: list = field(default_factory=list)
    plan: str = ""
    latency_ms: float = 0.0
    metrics: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    confidence: float | None = None
    errors: list = field(default_factory=list)


class ManagerAgent(BaseAgent):
    name: ClassVar[str] = "manager"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "总调度 Agent"
    layer: ClassVar[str] = "orchestration"
    depends: ClassVar[list[str]] = []
    Input: ClassVar[type] = dict     # the raw request dict (kind, source, ...)
    Output: ClassVar[type] = ManagerOutput

    def __init__(self) -> None:
        super().__init__()

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        if not raw_input.get("kind"):
            return False, "request.kind is required (cash_hand|tournament_hand|replay_hand)"
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> ManagerOutput:
        registry = getattr(ctx, "registry", None) or register_all_agents()
        runner = PipelineRunner(registry, ctx.config, ctx.logger, ctx.metrics)
        try:
            plan = build_plan(str(inp.get("kind")), ctx.config)
        except OrchestrationError as e:
            return ManagerOutput(kind=str(inp.get("kind", "?")), status="error",
                                 warnings=[{"code": "PLAN_ERROR", "message": str(e)}])
        report = await runner.run(inp, ctx)

        out = ManagerOutput(
            kind=report.kind, status=report.status,
            final_decision=report.final_decision,
            base_decision=report.base_decision,
            deviated_decision=report.deviated_decision,
            deviation_applied=report.deviation_applied,
            agents_run=[a for a in report.envelopes],
            agents_skipped=report.skipped,
            agents_degraded=report.degraded,
            plan=plan_as_doc(plan),
            latency_ms=report.latency_ms,
            metrics=report.metrics,
        )
        if report.degraded:
            out.warnings.append({"code": "DEGRADED_AGENTS", "message": ", ".join(report.degraded)})
        if report.deviation_applied and report.deviated_decision:
            out.warnings.append({
                "code": "DEVIATION_APPLIED",
                "message": f"策略偏离触发：原建议 {report.deviated_decision.get('original_action')} "
                           f"-> 最终 {report.deviated_decision.get('final_action')}",
            })
        if report.final_decision:
            out.confidence = float(report.final_decision.get("confidence") or 0.0)
            out.evidence.append({"kind": "final_decision", "text": f"最终建议: {report.final_decision.get('action')}",
                                 "source": "judge+deviation", "weight": 1.0})
        return out
